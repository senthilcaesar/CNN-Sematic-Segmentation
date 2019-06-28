import numpy as np



SO = np.load('SO_BICEPS.npy')


jump = '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/test_BICEPS/jump.txt'
with open(jump) as f:
    jump_arr = f.read().splitlines()
    jump_arr = list(map(int, jump_arr))

cases = 'testcase.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

count = 0
start = 0
end = start + jump_arr[0]

for i in range(0, len(jump_arr)):
	print(start, end, jump_arr[i])

	casex = SO[start:end,:,:]
	
	np.save(str(case_arr[i])+'/predict-gaussian.npy', casex)

	start = end
	end = end + jump_arr[i+1]
	#print("Case ", count, "done")
	count += 1
