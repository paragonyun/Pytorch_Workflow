## Pytorch Trouble Shooting
### 1. OOM
- 왜 / 어디서 발생했는지 알기 어려움
- Error Backtracking이 이상한 곳으로 감
- 메모리 이전상황의 파악을 하기가 어렵다.
    - 보통 iter를 돌면서 발생을 하는데, 돌다가 발생하면 그 전의 상황을 잘 모른다는 내용


> (1) 배치사이즈 줄이기
> (2) GPU Clearn (껐다 키기)
> (3) Run


코드만 잘 짰으면 이 방법이 제일 쉽고 편한 방법이긴 함

<br>

### 2. GPU(cuda) 
- GPUUtil사용하기
    - nvidia-smi 처럼 GPU의 상태를 보여주는 모듈이 있음
    - Colab은 애초에 GPU상태를 보여줌
    ```python
    !pip install GPUtil

    import GPUtil
    GPUtil.showUtilization()
    ```
    - Iter마다 메모리가 늘어나는지 확인하기

- `torch.cuda.empty_cache()` 써보기
    - 사용되지 않는 GPU상의 cache를 정리
    - 가용 메모리 확보
    - del 과는 다름 (del은 관계를 끊어줌) [사실상 안 줄어듬]
    - reset 대신 쓰기 좋은 함수
    - 항상 쓸 수는 없고 Loop 돌기 전에 한번 청소해주는 걸 권장

- Training Loop에 Tensor로 축적되는 변수를 확인해보기
    - Tensor로 처리되는 변수는 GPU로 올라가는데, 항상 이게 메모리를 잡아먹음
    - Backward를 하다보면 Total Loss를 계산하기 위해 각 단계당 loss를 저장해야됨
    - 이러면 gpu에 계속해서 쌓이게 되므로 비효율적 
        - 그래서 loss에 loss.item()을 사용해서 파이썬 기본 객체로 변환해주는 것!!!!!!

- del 명령어를 적절히 사용하기
    - for loop이 끝나도 내부에서 사용한 변수는 사용 가능
    - 그럼 이 변수의 메모리가 좀 크면 많이 비효율적이됨! 이런 건 del과 gc로 삭제하자!

- 가능한 Batch_Size를 실험해보기
    - Batch 사이즈를 1로 해서 실험해보기
    - 어디까지 가능한지 test 해보기

- inference를 할 땐 `torch.no_grad()`를 사용할 것
    - 메모리에 역전파를 쌓아두는 걸 방지함
    - 습관적으로 해두면 좋음

<br>

## 그 외...
- Colab에서 너무 큰 사이즈는 실행하지 말 것
    - Linear, CNN, LSTM 같은 건 실험적으로만 쓰세여

- CNN에서의 에러는 크기가 안 맞아서 발생할 가능성이 높으니
    - `torchsummary`로 사이즈를 맞출 것

- tensor의 부동소수점을 16bit로 줄여도 됨
    - 하다하다 안 되면 이걸 해보셈








