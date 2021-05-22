  # Boostcamp AI Tech Stage 1 - Image Classification

  ## Report

  https://www.notion.so/rmsdud/Wrap-up-9f47d7e7b0964ce5beba2dd0fd5adae1

  ## 소개

  COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

  감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

  따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

  ### 분류 방법

  - 마스크 착용 여부 : 착용 / 잘못된 착용 / 미착용
  - 성별 : 남 / 여
  - 연령 : 30대 미만 / 30대 이상 ~ 60대 미만 / 60대 이상

  총 18개의 label 분류

  ## 모델 소개

  ### Backbone

  - `EfficientNet-b5` (https://github.com/lukemelas/EfficientNet-PyTorch) model fine-tuning
  - `ViT` (Visison Transformer)

  ### Loss

  - `F1 loss` + `Focal loss` (gamma = 5)

  ### Training time augmentaion

  - `Center crop` (384 * 384)
  - 대비 제한 적응 히스토그램 평활화(`CLAHE`: Contrast-limited adaptive histogram equalization)

  ### Optimizer

  - `AdamP`

  ### Tensorboard log
  
  </center><img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c08f6b4d-9d8d-4acf-9cfe-38d5fd6151da/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210522%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210522T172517Z&X-Amz-Expires=86400&X-Amz-Signature=80a01f71444bba4e2750c1b970ea3ce1e8551ab0d9843c9eb3eac45eb088dda0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width="450" height="450"></center>

  ## 모델 성능
  
  - F1 Score : 0.7660
  - Accuracy : 81.1111%
