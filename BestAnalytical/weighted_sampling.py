# Create new array : square of the sum of masses in dataset                          
sum_array = torch.zeros(len(train_Dataset), dtype=torch.float)

for i in range(len(train_Dataset)):
    sum_array[i] = torch.sum(sum(train_Dataset[i]))**2
    # [sum(sum(HI, DM))]^2 

#create list of weights
weights_train = [sum_array[x]/len(train_Dataset) for x in range(len(train_Dataset))]

#Scale weights
weights = weights_train / np.sum(weights_train)

#Use torch WeightedRandomSampler 
weighted_sampler = WeightedRandomSampler(
    weights=weights,                                                                                                        
    num_samples=len(weights),
    replacement=True
)

#Create data loader (Shuffle and weighted_sampler are mutually exclusive)
train_loader = DataLoader(dataset=train_Dataset, shuffle=False,
                          batch_size=batch_size, sampler=weighted_sampler)