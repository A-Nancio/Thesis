
from distutils.command.sdist import sdist
from os import dup
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv('uids_v4_no_multiuid_cleaning..csv', index_col='TransactionID')

#original dataset
dups_color = pd.read_csv('banksim/DatasetWork/bankSimDataClean.csv')
total_trans = dups_color.shape[0]
print(total_trans)
n_col = len(dups_color.columns)
customers = dups_color.customer.unique()
genders = dups_color.gender.unique()
ages = dups_color.age.unique()

print(customers)
print(ages)
#print(customers[0])

#df = dups_color.pivot_table(columns=['uid','count'], aggfunc='size')
# print (dups_color)
# print(dups_color.describe())

def Gender_Age_Handle():
    #print(dups_color['age'].value_counts())
    #global dups_color
    #dups_color = dups_color[dups_color['age'] != "'U'" ]
    #Categorical('gender')
    Age('age')

def CustomerId():
    CustomerMapping('customer')
    

# get table with frequency for each column - dff
def frequencyForColumn():    
    for i in range(n_col):
        col_name = dups_color.columns[i]
        print(col_name)
        dff = dups_color.iloc[:,i].value_counts(sort=True).rename_axis(col_name).to_frame('Sequence Size')
        #dff.columns = [col_name, 'Sequence Size']
        print (dff)
        print(dff.describe())
        print(dff.columns)
        #dff.to_csv('banksim/' + col_name + 'Frequency.csv')
        if col_name == 'gender' or col_name == 'age':
            axarr = dff.plot(kind='bar')
            # for ax in axarr.flatten():
            #     ax.set_xlabel("size of sequence")
            #     ax.set_ylabel(col_name)
        else:
            axarr = dff.hist(column='Sequence Size')
            for ax in axarr.flatten():
                ax.set_xlabel("size of sequence")
                ax.set_ylabel(col_name)
        plt.title(col_name)
        plt.show()
    plt.figure()

def Barchart():
    dff = pd.read_csv('banksim/ColFrequencies/' + 'merchant' + 'Frequency.csv')
    
    axarr = dff.plot(kind='bar')
    # for ax in axarr.flatten():
    #     ax.set_xlabel("size of sequence")
    #     ax.set_ylabel(col_name)
    
    plt.title('merchant')
    plt.show()

def General():
# #hisptogram for step, amount and fraud 
# ------------------
    dups_color.hist(column='step', bins=180)
    plt.show()
#-----------------------

#amount <200 
# ------------------
# dff = dups_color[dups_color['amount'] < 200]
# dff.hist(column='amount', bins=20)
# plt.show()
#-----------------------

#count unique values for each column 
# -----------
# 
# print(n_col)
# f = open('Unique_Values_count.txt', 'w')
# for i in range(n_col):
#     count = dups_color.iloc[:,i].nunique()
#     print("column name:" + dups_color.columns[i] +", count:" + str(count) )
#     f.write("column name:" + dups_color.columns[i] +", count:" + str(count) + "\n")
#-------------------------

#Separate card sequences to individual csv
# ---------------

#res = dups_color[dups_color['customer'] == customers[0]]
# res.drop(['Unnamed: 0'], axis=1, inplace = True)
# res.to_csv('banksim/DatasetWork/Customers/client.csv')
#print(res)
# for x in customers:
#     res = dups_color[dups_color['customer'] == x]
#     res.drop(['Unnamed: 0'], axis=1, inplace = True)
#     res.to_csv('banksim/DatasetWork/Customers/'+ x +'.csv')
#-----------------------

#calculate new column with distance in time for each customer

# for x in customers:
#     cus = pd.read_csv('banksim/DatasetWork/Customers/' + x + '.csv')
#     n_rows = cus.shape[0]
#     print(n_rows)
#     new_col = [0]* n_rows
#     prev_step = 0
#     for index, row in cus.iterrows():
#         #print(index)
#         step = row['step']
#         if index == 0:
#             prev_step = step
#             continue
#         new_col[index] = step - prev_step
#         prev_step = step
#         #print(row)

#     cus['distance'] = new_col

#     #cus = cus.iloc[: , 1:]
#     #print(cus)
#     cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)
#     axarr = cus.hist(column='distance')
#     for ax in axarr.flatten():
#         ax.set_xlabel("distancia")
#         ax.set_ylabel("frequency")
#     plt.title(x)
#     #plt.show()
#     plt.savefig('banksim/DatasetWork/DistanceHisptogram/' + x + '.png')
#     plt.close()
#-----------------------


# concatenate all customers in order to do a general hisptograms about distances
#---------------
# frames = []
# for x in customers:
#     cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
#     frames = frames + [cus]
# result = pd.concat(frames)
# #dff = result[result['distance'] < 10]
# axarr = result.hist(column='distance', bins=120)
# for ax in axarr.flatten():
#     ax.set_xlabel("distancia")
#     ax.set_ylabel("frequency")
# plt.title("Distance between step by custumer")
# plt.show()
#plt.savefig('banksim/DatasetWork/DistanceHisptogram/' + x + '.png')
#plt.close()
#-----------------------


# #  z scoring in numerical variables: amount
def zScoringNumerical():
    amount_mean = dups_color['amount'].mean()
    amount_std = dups_color['amount'].std(ddof=0)
    frames = []
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        frames = frames + [cus]
    result = pd.concat(frames)
    distance_mean = result['distance'].mean()
    distance_std = result['distance'].std()
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        col_zscore = 'amount_zscore'
        cus[col_zscore] = (cus['amount'] - amount_mean)/amount_std
        col_zscore2 = 'distance_zscore'
        cus[col_zscore2] = (cus['distance'] - distance_mean)/distance_std
        cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)
# #---------------

def GenderCustomers():
    data = {}
    for x in genders:
        dff = dups_color[dups_color['gender'] == x]
        i = len(dff.customer.unique())
        data[x] = i
        print(data)
    genderss = list(data.keys())
    Ncustomers = list(data.values())
    
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(genderss, Ncustomers, color ='maroon',
            width = 0.4)
    
    plt.xlabel("Genders")
    plt.ylabel("No. unique customers")
    plt.title("Unique customers by gender")
    plt.show()

def Categorical(feature):
    dt = pd.read_csv('banksim/ColFrequencies/' + feature + 'Frequency.csv')
    #dt = dt.sort_values(by=['Sequence Size'])
    cat = dt[feature].tolist()
    print(cat)
    print(cat[1])
    feature_mapped = feature + 'mapped'
    j = 1
    for x in customers:
        print(j)
        Type_new = pd.Series([])
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        N = len(cus)
        for i in range(N):
            value = cat.index(cus[feature][i])
            #print(str(cus['category'][i]) + '---' + str(value))
            Type_new[i] = value
        cus[feature_mapped] = Type_new
        #print(cus)
        cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)
        j = j+1

def Age(feature):
    dt = pd.read_csv('banksim/ColFrequencies/' + feature + 'Frequency.csv')
    #dt = dt.sort_values(by=['Sequence Size'])
    cat = dt[feature].tolist()
    print(cat)
    print(cat[1])
    feature_mapped = feature + 'mapped'
    j = 1
    for x in customers:
        print(j)
        Type_new = pd.Series([])
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        
        N = len(cus)
        for i in range(N):
            # value = cat.index(cus[feature][i])
            # #print(str(cus['category'][i]) + '---' + str(value))
            if cus[feature][i] == "'U'":
                value = -1
            else:
                value = int(cus[feature][i][1])
            Type_new[i] = value
        cus[feature_mapped] = Type_new
        #print(cus)
        cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)
        j = j+1

def CustomerMapping(feature):
    dt = pd.read_csv('banksim/ColFrequencies/' + feature + 'Frequency.csv')
    #dt = dt.sort_values(by=['Sequence Size'])
    cat = dt[feature].tolist()
    print(cat)
    print(cat[1])
    feature_mapped = feature + 'Id'
    j = 1
    for x in customers:
        print(j)
        Type_new = pd.Series([])
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        
        N = len(cus)
        for i in range(N):
            value = int(cus[feature][i][2:-1])
            #print(str(cus['category'][i]) + '---' + str(value))
            Type_new[i] = value
        cus[feature_mapped] = Type_new
        #print(cus)
        cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)
        j = j+1

def CategoricalMerchant():
    dt = pd.read_csv('banksim/ColFrequencies/merchantFrequency.csv')
    #dt = dt.sort_values(by=['Sequence Size'])
    cat = dt['merchant'].tolist()
    print(cat)
    print(cat[1])

    for x in customers:
        Type_new = pd.Series([])
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        N = len(cus)
        for i in range(N):
            value = cat.index(cus['merchant'][i])
            #print(str(cus['category'][i]) + '---' + str(value))
            Type_new[i] = value
        cus['merchant_mapped'] = Type_new
        #print(cus)
        cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)

def StepStudy():
    for x in customers:
        Type_new = pd.Series([])
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        N = len(cus)
        val = 0
        last = -1
        value = val
        count = 1
        rep = 0
        for i in range(N):
            val = cus['step'][i]
            if last == val:
                value = val + count*0.1
                count = count + 1
                rep = rep + 1
                if count == 10:
                    print('-------dsds--------'+ str(i))

            else:
                value = val
                count = 1
                rep = 0
            Type_new[i] = value
            last = val
        cus['New Step'] = Type_new
        cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)

# def StepOperation():
#     for x in customers:
#         Type_new = pd.Series([])
#         cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
#         N = len(cus)
#         val = 0
#         last = -1
#         value = val
#         count = 1
#         rep = 0
#         for i in range(N):
#             val = cus['step'][i]
#             #print(str(cus['category'][i]) + '---' + str(value))

#             if last == val:
#                 value = val + count*0.2
#                 count = count + 1
#                 rep = rep + 1
#                 if count == 5:
#                     print('-------dsds--------'+ str(i) + '---'+ x)

#             else:
#                 if rep >= 1:
#                     u = 0
#                     cof = 1/(count+1)
#                     while u < rep:
#                         Type_new[i-u] = val + cof*

#                 value = val
#                 count = 1
#                 rep = 0
#             #Type_new[i] = value
#             last = val
#         cus['New Step'] = Type_new
#         #print(cus)
#         cus.to_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv', index =False)

# # axarr = dups_color.hist(column='counter', bins=100)
# for ax in axarr.flatten():
#     ax.set_xlabel("size of sequence")
#     ax.set_ylabel("number of cards")
# plt.show()
# dff = dups_color[dups_color['counter'] > 5]
# dff2 = dff[dff['counter'] < 500]
# print(dff2)
# axarr2 = dff2.hist(column='counter', bins=100)
# for ax in axarr2.flatten():
#     ax.set_xlabel("size of sequence")
#     ax.set_ylabel("number of cards")
# plt.show()
# #dups_color.hist(column='uid')
# #plt.show()
# #dups_color.to_csv('uidFrequency.csv')


def divideDataset():
    fraudCus = False
    count = 0
    cusFraud = []
    cusNoFraud = []
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        N = len(cus)
        for i in range(N):
            val = cus['fraud'][i]
            if val == 1:
                fraudCus = True
                print(x + ' customer name , position '+ str(i))
                count = count + 1
                cusFraud = cusFraud + [x]
                #cus.to_csv('banksim/DatasetWork/Divided/WithFraud/' + x + '.csv', index =False)
                break
        if fraudCus:
            fraudCus = False
            continue 
        cusNoFraud = cusNoFraud + [x]
        #cus.to_csv('banksim/DatasetWork/Divided/WithNoFraud/' + x + '.csv', index =False)
    f = open('CustomersWithFraud.txt', 'w')
    for x in cusFraud:
        f.write(x + "\n")
    
    g = open('CustomersNoFraud.txt', 'w')
    for x in cusNoFraud:
        g.write(x + "\n")
        
    print('number of cards with fraud:' + str(count))

def Train_Validation_Test_Split():
    Steps = 180
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/CustomersDistance/' + x + '.csv')
        N = len(cus)
        #Train from 0 to 120
        train = cus[cus['step'] < 120]
        train.to_csv('banksim/DatasetWork/Divided/Train/' + x + '.csv', index =False)
        #validation from 120 to 150
        validation = cus[cus['step'] >= 120]
        validation = validation[validation['step'] < 150]
        validation.to_csv('banksim/DatasetWork/Divided/Validation/' + x + '.csv', index =False)
        #Test from 150 to 180
        test = cus[cus['step'] >=150]
        test.to_csv('banksim/DatasetWork/Divided/Test/' + x + '.csv', index =False)

def divideDatasetTrain():
    fraudCus = False
    count = 0
    cusFraud = []
    cusNoFraud = []
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/Divided/Train/' + x + '.csv')
        N = len(cus)
        for i in range(N):
            val = cus['fraud'][i]
            if val == 1:
                fraudCus = True
                print(x + ' customer name , position '+ str(i))
                count = count + 1
                cusFraud = cusFraud + [x]
                cus.to_csv('banksim/DatasetWork/Divided/Train/Fraud/' + x + '.csv', index =False)
                break
        if fraudCus:
            fraudCus = False
            continue 
        cusNoFraud = cusNoFraud + [x]
        cus.to_csv('banksim/DatasetWork/Divided/Train/NoFraud/' + x + '.csv', index =False)
    f = open('banksim/DatasetWork/Divided/Train/Fraud/CustomersWithFraud.txt', 'w')
    for x in cusFraud:
        f.write(x + "\n")
    
    g = open('banksim/DatasetWork/Divided/Train/NoFraud/CustomersNoFraud.txt', 'w')
    for x in cusNoFraud:
        g.write(x + "\n")
        
    print('number of cards with fraud:' + str(count))

def divideDatasetValidation():
    fraudCus = False
    count = 0
    cusFraud = []
    cusNoFraud = []
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/Divided/Validation/' + x + '.csv')
        N = len(cus)
        for i in range(N):
            val = cus['fraud'][i]
            if val == 1:
                fraudCus = True
                print(x + ' customer name , position '+ str(i))
                count = count + 1
                cusFraud = cusFraud + [x]
                cus.to_csv('banksim/DatasetWork/Divided/Validation/Fraud/' + x + '.csv', index =False)
                break
        if fraudCus:
            fraudCus = False
            continue 
        cusNoFraud = cusNoFraud + [x]
        cus.to_csv('banksim/DatasetWork/Divided/Validation/NoFraud/' + x + '.csv', index =False)
    f = open('banksim/DatasetWork/Divided/Validation/NoFraud/CustomersWithFraud.txt', 'w')
    for x in cusFraud:
        f.write(x + "\n")
    
    g = open('banksim/DatasetWork/Divided/Validation/NoFraud/CustomersNoFraud.txt', 'w')
    for x in cusNoFraud:
        g.write(x + "\n")
        
    print('number of cards with fraud:' + str(count))

def divideDatasetTest():
    fraudCus = False
    count = 0
    cusFraud = []
    cusNoFraud = []
    for x in customers:
        cus = pd.read_csv('banksim/DatasetWork/Divided/Test/' + x + '.csv')
        N = len(cus)
        for i in range(N):
            val = cus['fraud'][i]
            if val == 1:
                fraudCus = True
                print(x + ' customer name , position '+ str(i))
                count = count + 1
                cusFraud = cusFraud + [x]
                cus.to_csv('banksim/DatasetWork/Divided/Test/Fraud/' + x + '.csv', index =False)
                break
        if fraudCus:
            fraudCus = False
            continue 
        cusNoFraud = cusNoFraud + [x]
        cus.to_csv('banksim/DatasetWork/Divided/Test/NoFraud/' + x + '.csv', index =False)
    f = open('banksim/DatasetWork/Divided/Test/Fraud/CustomersWithFraud.txt', 'w')
    for x in cusFraud:
        f.write(x + "\n")
    
    g = open('banksim/DatasetWork/Divided/Test/NoFraud/CustomersNoFraud.txt', 'w')
    for x in cusNoFraud:
        g.write(x + "\n")
        
    print('number of cards with fraud:' + str(count))
# General()
# Barchart()
#GenderCustomers()
#CategoricalMerchant()
#StepOperation()
#StepStudy()
#Gender_Age_Handle()
CustomerId()
Train_Validation_Test_Split()
divideDatasetTrain()
divideDatasetValidation()
divideDatasetTest()