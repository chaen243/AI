import numpy as np
import hyperopt
import pandas as pd
print(hyperopt.__version__) # 0.2.7

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

search_space = {'x1' : hp.quniform('x1', -10, 10, 1),
                'x2' : hp.quniform('x2', -15, 15, 1)}
                # hp.quniform(label, low, high, q)
                #단위 안에서 균등하게 배포

# hp.quniform(label, low, high, q) : label로 지정된 입력 값 변수 검색 공간을 
#                                    최소값 low에서 최대값 high까지 q의 간격을 가지고 설정

# hp.uniform(label, low, high, q) : 최소값 low에서 최대값 high까지 정규분포 형태의 검색 공간 설정

# hp.randint(label, upper): 0부터 최대값 upper까지 random한 정수값으로 검색 공간 설정.

# hp.loguniform(label, low, high) : exp(uniform(low, high))값을 반환하며, 
#                                   반환값의 log변환 된 값은 정규분포 형ㅌ를 가지는 검색공간 설정.

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value =  x1**2 -20*x2
    #x1이 0이면 최소값
    #x= 15이면 최대값.    
    
    return return_value
    
trial_val = Trials()

best = fmin(
    fn = objective_func,
    space= search_space,
    algo = tpe.suggest, #디폴트
    max_evals=20, #서치 횟수
    trials= trial_val,
    rstate=np.random.default_rng(seed=10),
    #rstate= 10
)    

print(best) #{'x1': 0.0, 'x2': 15.0}

print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'}, {'loss': 200.0, 'status': 'ok'}, 
# {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}, {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, 
# {'loss': -11.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, 
# {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'}, {'loss': -300.0, 'status': 'ok'}, 
# {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]

print(trial_val.vals)
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 
#  'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}



print('|   iter   |  target  |    x1    |    x2    |')
print('---------------------------------------------')
x1_list = trial_val.vals['x1']
x2_list = trial_val.vals['x2']

for idx, data in enumerate(trial_val.results ):
    loss = data['loss']
    print(f'|{idx+1:^10}|{loss:^10}|{x1_list[idx]:^10}|{x2_list[idx]:^10}|')    
    #{loss:^10} = 10칸중에 loss값을 가운데 정렬
    
# |   iter   |  target  |    x1    |    x2    |
# ---------------------------------------------
# |    1     |  -216.0  |   -2.0   |   11.0   |
# |    2     |  -175.0  |   -5.0   |   10.0   |
# |    3     |  129.0   |   7.0    |   -4.0   |
# |    4     |  200.0   |   10.0   |   -5.0   |
# |    5     |  240.0   |   10.0   |   -7.0   |
# |    6     |  -55.0   |   5.0    |   4.0    |
# |    7     |  209.0   |   7.0    |   -8.0   |
# |    8     |  -176.0  |   -2.0   |   9.0    |
# |    9     |  -11.0   |   -7.0   |   3.0    |
# |    10    |  -51.0   |   7.0    |   5.0    |
# |    11    |  136.0   |   4.0    |   -6.0   |
# |    12    |  -51.0   |   -7.0   |   5.0    |
# |    13    |  164.0   |   -8.0   |   -5.0   |
# |    14    |  321.0   |   9.0    |  -12.0   |
# |    15    |   49.0   |   -7.0   |   0.0    |
# |    16    |  -300.0  |   0.0    |   15.0   |
# |    17    |  160.0   |   -0.0   |   -8.0   |
# |    18    |  -124.0  |   4.0    |   7.0    |
# |    19    |  -11.0   |   3.0    |   1.0    |
# |    20    |   0.0    |   -0.0   |   0.0    |

target = [aaa['loss'] for aaa in trial_val.results]
print(target)
df = pd.DataFrame({'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2']
                   })
print(df)