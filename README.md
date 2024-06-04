# GenerativeModel_Tobigs_Conference
###### 모든 코드는 용진이형의 코드로 이에 대한 설명과 DIffusion 코드 설명이 있습니다.
###### 정말 아무것도 모르는 통계쟁이가 쓴 것이어서 정보의 오류가 있을 수 있습니다.

# File Structure

```
StableDiffusion
├── configs/ # config 를 넣어놓은 파일 .yaml로 전달하여 다양한 실험 설정 한번에 관리
│   │
│   ├── dataset/
│   │   └── textual_inversion_dataset.yaml #Dataloader와 관련된 config 저장
│   │
│   ├── model/
│   │   └── diffusion
│   │       └── sdlx_ldm # Baseline / VAE / Scheduler 등 설정
│   │   └── Tokenizer
│   │       └── sdxl_tokenizer # Input을 인코딩할 모델과 관련된 설정
│   │   
│   ├── trainer/
│   │   └── textual_inversion_trainer.yaml # optimizer와 같은 딥러닝 학습 도구
│   └── base.yaml # 실험 설계 wandb와 같은 툴 사용시 필요
│   │   
│   └── TextualInversion.yaml  # 한 번에 다 부르기 __init__ 같은 존재
│   
│   
├── src/ # 실질적 모델이 있는 파일 - Diffusers 기반의 모델
│   │   
│   ├── common/
│   │   └── logger.py # wandb와 같은 실험 기록 툴을 위한 파일 
│   │   └── schedulers.py # LR을 cosineAnnealing 방법으로 주기적 + WarmUp 
│   │   └── train_utils.py # Seed 설정 및 Placeholder 설정 
│   │ 
│   ├── datasets/
│   │   └── __init__.py # 추후 datasets.build_dataloader를 이용해 dataloader 생성
│   │   └── Dreambooth.py # 동물 부분만 크롭하고 Dreambooth 적용하는 코드로 추정,,,(?)
│   │   └── textulainversion.py # 동물 부분만 크롭하고 Text 임베딩 용 파일
│   │  
│   └──── models/
│       ├── __init__.py # 전체 Diffusion Model 빌드 : config에서 모델 설정하여 SD 생성
│       │
│       ├── adapter
│       │    └── Lora.py # LoRA 실행 파일 - 기존 모델에 가중치 업데이트 시키는 파일 - 
│       ├── diffusion # 디퓨전이다
│       │    └── __init__.py # get_ldm_model()로 바로 SD모델 불러올 수 있음
│       │    └── diffusion_scheduler.py # 노이즈를 조절하는 스케쥴러 불러오기 
│       │    └── stable_diffusion.py # 메인모델 부분 - Diffuers 사용
│       │
│       ├── tokenizer
│       │    └── __init__.py # get_tokenizer_model()로 CLIP 임베딩 모델 사용
│       │    └── CLIP.py # Transformers에서 CLIP encoder 설정
│       │
│       └── tokenizer
│            └── __init__.py # 
│            └── train_text_inversion.py # 모델 weights freeze / 훈련환경 설정 / 훈련 진행
│   
├── run.py # 전체 훈련 및 이미지를 통해 원하는 이미지 생성 
└── vis.ipynb # Inference 코드가 담긴


```



## 디퓨전 모델 구조
- 기본 코드는 다음과 같다
``` python
import torch
from diffusers import StableDiffusionPipeline

# 모델 불러오기
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  

# 이미지 생성
prompt = "A fantasy landscape with mountains and a river"
image = pipe(prompt).images[0]


```
- SD model은 Text-Prompt를 받아서 이미지를 생성하는 Text-to-Image 모델
- 이렇게 단순하게 하면 원하는 결과를 얻기가 힘들다
- SD는 크게 3가지로 구성되어있다 -> [VAE / U-Net / Text Encoder]
- 각 모델을 간단하게 요약하면 (LDM 기준)
    - VAE : Diffusion 에서 나온 $z$를 이미지를 고해상도로 변경 (픽셀화)
    - U-Net : Diffusion Process에서 노이즈 $z_T$에서 점차 $z_0$으로 Denoising
    - CLIP : 원래 용도는 이미지와 텍스트를 모두 처리할 수 있게 만들어졌지만 텍스트 임베딩 도구로 사용


- 이를 활용하여 만든 베이스라인은 다음과 같다
``` python
from diffusers import StableDiffusionPipeline

# 파이프라인 설정
pipe = StableDiffusionPipeline(
    vae=AutoencoderKL.from_pretrained("path/to/save/vae"),
    text_encoder=CLIPTextModel.from_pretrained("path/to/save/text_encoder"),
    tokenizer=CLIPTokenizer.from_pretrained("path/to/save/tokenizer"),
    unet=UNet2DConditionModel.from_pretrained("path/to/save/unet"),
    scheduler=DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
)
pipe = pipe.to(device)

# 이미지 생성
prompt = "A beautiful sunset over the mountains"
image = pipe(prompt).images[0]

```

- Diffusers의 StableDiffusionPipeline을 사용하는데 이는 생성 모델을 일련과정으로 표현해주는 것
``` python
import torch
from torch import nn, optim

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae.to(device)
text_encoder.to(device)
unet.to(device)

optimizer = optim.Adam(unet.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# 훈련 루프
for epoch in range(10):  # 에포크 수
    for batch in dataloader:
        images = batch["images"].to(device)
        texts = batch["texts"]
        
        # 텍스트 인코딩
        text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        encoder_hidden_states = text_encoder(text_inputs).last_hidden_state

        # 이미지 인코딩
        latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Diffusion 단계 설정
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = scheduler.add_noise(latents, torch.randn_like(latents), timesteps)

        # 모델 예측
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 손실 계산
        loss = loss_fn(noise_pred, latents)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")

```

- 위와 같은 과정으로 U-Net을 학습시킨다 (주로 VAE와 텍스트 인코더는 고정, 그렇기 떄문에 실질적으로 우리가 Fine-tuning 혹은 Downstream Task에서는 Stable Diffusion은 U-Net을 학습한다고 생각하면 됨 )
- 그렇기 때문에 Adatper와 같은 류들은 Condition 등을 조절하여 U-Net을 조절









# Stable Video Diffusion

- Baseline은 Video LDM이다
    - Video LDM : 이미지 LDM을 구성하는 레이어의 차원 입력을 공간적 레이어로 처리
    - 베이스 모델은 Text-to-Image 모델이다
    - 시간에 대한 감각이 없기에 시간적인 레이어 추가 
    - $z \in \mathbb{R}^{T \times C \times H \times W}$ 의 데이터
    - 시간 차원까지 배치로 처리해 (B * T) C H W 로 적용
    - Temporal Layer에 적용하기 위하여 Shape를 변경 (b c t h w)
    - 이는 곧 T 차원에서 처리를 통해 시간적인 정보를 사용
    - Temporal Layer를 Temporal Mixing layer로 구현하여 하나는 3D convolution / attention block 으로 구성
    - 레이어를 나온 출력을 바로 쓰는 것이 아닌 $\alpha^i_{\phi} z + (1- \alpha^i_{\phi})z^{'} ; \alpha^i_\phi \in [0,1]$ 로 학습
    - 공간적 레이어는 고정하고 시간적 레이어만 최적화
    - $\argmin_\phi \mathbb{E}_{x\sim p_data, \tau \sim p_r, \epsilon \sim \mathcal{N}(0,I)}[\vert\vert y - f_{\phi,\theta}(z_\tau;c,\tau)\vert\vert^2_2] $ -> 시간적 레이어에 대해서 인코딩 된 $z$의 MSE를 학습
    - Temporal Autoencoder Finetuning : T2I의 오토인코더 또한 이미지 기반이어서 오토인코더의 디코더 부분에 추가적인 temporal lyaers를 적용
    - 전체적인 프로세스 
        1. Key Frame을 생성해 핵심 프레임 생성
        2. 프레임을 이용해 보간을 수행 -> FPS를 늘리는 작업
        3. 보간 완료한 latent vector에 대한 디코딩 수행
        4. Upsampler를 이용하여 Super Resolution 수행
     
![Architecture](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzQ1zN%2FbtsGKWIkky1%2FYoEnYpx1wkvkG5FNPKsSIk%2Fimg.png)
![Process](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Flc7St%2FbtsGKuL6vYf%2Fk3qXKMWCPxYQ3OePyRK651%2Fimg.png)


- Stable Video Diffusion에서는 비슷하게 사용
    - Image 모델 학습 / 이미지 모델을 비디오로 확장 / 미세조정
    - Diffusers를 사용시 [UNetSpatioTEmporalConditionModel / AutoencoderKLTemporalDecoder / CLIPVisionModelWithProjection / EulerDiscreteScheduler / CLIPImageProcessor] 로 모델이 크게 구성
- Stable Video Diffusion을 불러와서 학습 시키는 코드를 만드려했는데 지금 cpu밖에 없는 상태라 하진 못하였지만 비슷할 것으로 예상합니다. 이번주 내로 한 번 돌려보고 추가해볼게요 :)
