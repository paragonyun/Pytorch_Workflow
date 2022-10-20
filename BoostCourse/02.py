'''
미분에 대하여

epochs = 10
for epoch in range(epochs) :
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, labels)
    print(loss)

    loss.backward()

    optimizer.step()

여기선 backward와 step 단계를 구현해볼거임
https://www.boostcourse.org/ai213/lecture/1418317/?isDesc=false
'''

'''
backward는 Module 단계에서 직접 지정할 수 있음
Module에서 오버라이딩 시키면 됨
근데 이거 좀 부담스러우니까 쓰지는 않을 거 같긴 한데 순서는 이해하자
'''







