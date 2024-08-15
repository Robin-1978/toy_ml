import DataModel




red_train, red_target, blue_train, blue_target = DataModel.PrepareSSQ(3)

print(red_train.shape, red_target.shape, blue_train.shape, blue_target.shape)