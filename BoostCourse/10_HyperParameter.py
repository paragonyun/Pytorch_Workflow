'''
1. 보통 가장 먼저 모델을 바꿔봄
2. 데이터를 바꿔봄 - 추가하거나 오류가 없는지 찾아보거나
3. 하이퍼파라미터 튜닝
-- 이 중에서 가장 좋은 성능을 내는 건 2번!
-- 1번은 이미 좋은 모델은 알고있고 고정 되어 있음 -> 그렇게 큰 변화폭은 없음
-- 하이퍼 파라미터는 그다지 큰 성능 향상은 없다이!!

사람이 지정하는 값
- LR, 모델크기, optimizer 등 (NAS는 lr도 알아서 찾아줌)
- 가끔은 파라미터에 의해 값이 바뀔 때도 있는데 요즘은 그닥
- 0.01을 쥐어 짜야할 때 해볼만 하다.

'''
## Grid Search
# 모든 값들을 테스트 해보는 기법 
# 일정한 범위를 통해 자르는 기법 (일정한 비율로)

## Random Search
# 값들을 램덤하게 찾아보는 기법
# 실제로는 Random으로 하다가 잘 나오는 구간이 있으면 거기서 Grid로 해보긴 함
# 지금은 뭐.. 잘 안 하긴 함
# 지금은 베이지안 옵티마이제이션을 많이 함 (BOHB)

## ❗❗ Ray ❗❗ 
# Multi-node multi processing 지원 모듈
# => 여러 대의 컴퓨터를 연결해서 지원함
# ML/DL의 병렬 처리를 위해 개발된 모듈
# 분산병렬의 표준이다!
# Spark를 만든 연구실에서 만든 곳임
# Hyperparameter Search를 위한 다양한 모듈 제공
import os

data_dir = os.path.abspath('./data')
load_data(data_dir)

config = {
    'l1' : tune.sample_from(lambda _ : 2**np.random.randint(2,9)),
    'l2' : tune.sample_from(lambda _ : 2**np.random.randint(2,9)),
    'lr' : tune.loguniform(1e-4, 1e-1), ## grid
    'batch_size' : tune.choice([2,4,8,16]) ## grid
} ## l1과 l2는 그냥 마지막 레이어들의 크기를 말하는 거

## 이 스케쥴러 한번 써보기 (Ray에 있음)
scheduler = ASHAScheduler(
    metric='loss', mode='min', max_t=max_num_epochs, grace_period=1, reduction_factor=2
)

## Reporter, 아래 꺼는 Command Line을 출력하는 곳
reporter = CLIReporter(
    metric_columns=['loss','accuracy','training_iteration']
)

result = tune.run(
    partial(train_cifar, data_dir=data_dir),
    resources_per_trial={'cpu':2, 'gpu':gpus_per_trial},
    config=config, num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter
)

'''
주의할 점은
RAy를 활용하기 위해선 학습 과정이 모두 하나의 "함수"에 들어가 있어야함
'''
























