from genericpath import exists
import pandas as pd
import matplotlib.pyplot as plt
from zmq import has
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

def buildDifBatches(cutoff_lenght):
    """ batch size 24"""
    batch_size = 24

    train, n_batches1 = batchConstruction4(batch_size,cutoff_lenght, 'Train')
    train.to_csv('banksim/batches24.csv', index =False)

    #validation
    validation, n_batches_Val1 = batchConstruction4(batch_size,cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation24.csv', index =False)

    #test
    test, n_batches_test1 = batchConstruction4(batch_size,cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test24.csv', index =False)

    """batch size 12"""
    batch_size = 12

    train, n_batches2 = batchConstruction4(batch_size,cutoff_lenght, 'Train')
    train.to_csv('banksim/batches12.csv', index =False)

    #validation
    validation, n_batches_Val2 = batchConstruction4(batch_size,cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation12.csv', index =False)

    #test
    test, n_batches_test2 = batchConstruction4(batch_size,cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test12.csv', index =False)
    

    """batch size 6"""
    batch_size = 6

    train, n_batches3 = batchConstruction4(batch_size,cutoff_lenght, 'Train')
    train.to_csv('banksim/batches6.csv', index =False)

    #validation
    validation, n_batches_Val3 = batchConstruction4(batch_size,cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation6.csv', index =False)

    #test
    test, n_batches_test3 = batchConstruction4(batch_size,cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test6.csv', index =False)
    
    """batch size 1 or 2"""
    batch_size = 2

    train, n_batches4 = batchConstruction4(batch_size,cutoff_lenght, 'Train')
    train.to_csv('banksim/batches2.csv', index =False)

    #validation
    validation, n_batches_Val4 = batchConstruction4(batch_size,cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation2.csv', index =False)

    #test
    test, n_batches_test4 = batchConstruction4(batch_size,cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test2.csv', index =False)
    return train, validation, test

def buildSmallBatches(cutoff_lenght):
    train, n_batches4 = batchConstructionSmall(cutoff_lenght, 'Train')
    train.to_csv('banksim/batchesSmall.csv', index =False)

    #validation
    validation, n_batches_Val4 = batchConstructionSmall(cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validationSmall.csv', index =False)

    #test
    test, n_batches_test4 = batchConstructionSmall(cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_testSmall.csv', index =False)
    return train, validation, test

def build1Batches(cutoff_lenght):

    train, n_batches4 = batchConstruction1(cutoff_lenght, 'Train')
    train.to_csv('banksim/batches1Long.csv', index =False)

    #validation
    validation, n_batches_Val4 = batchConstruction1(cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation1Long.csv', index =False)

    #test
    test, n_batches_test4 = batchConstruction1(cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test1Long.csv', index =False)
    return train, validation, test

def buildBatchesSeq(cutoff_lenght):

    #datasetWorkTogether('Train')
    train, n_batches4 = batchConstruction6(cutoff_lenght, 'Train')
    train.to_csv('banksim/batches5SeqFull.csv', index =False)

    #validation
    #datasetWorkTogether('Validation')
    validation, n_batches_Val4 = batchConstruction6(cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation5SeqFull.csv', index =False)

    #test
    #datasetWorkTogether('Test')
    test, n_batches_test4 = batchConstruction6(cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test5SeqFull.csv', index =False)
    return train, validation, test

def buildTest(cutoff_lenght):    
    test, n_batches_test4 = batchConstruction1Test(cutoff_lenght, 'Test')
    test.to_csv('banksim/batches_test2clean.csv', index =False)

def buildTrainBatches(batch_size, cutoff_lenght):
    train, n_batches = batchConstruction4(batch_size,cutoff_lenght, 'Train')
    train.to_csv('banksim/batches.csv', index =False)
    return train

def buildBatches(batch_size,cutoff_lenght):
    #train
    train, n_batches = batchConstruction4(batch_size,cutoff_lenght, 'Train')
    train.to_csv('banksim/batches2.csv', index =False)

    #validation
    validation, n_batches_Val = batchConstruction4(batch_size,cutoff_lenght, 'Validation')
    validation.to_csv('banksim/batches_validation2.csv', index =False)

    #test
    # test, n_batches_test = batchConstruction4(batch_size,cutoff_lenght, 'Test')
    # test.to_csv('banksim/batches_test.csv', index =False)
    return train, validation#, test

def readSmallBatches():
    train = pd.read_csv('banksim/batchesSmall.csv')

    validation = pd.read_csv('banksim/batches_validationSmall.csv')

    test = pd.read_csv('banksim/batches_test1clean.csv')

    return train, validation, test

def readBatches():
    train = pd.read_csv('banksim/batches12.csv')

    validation = pd.read_csv('banksim/batches_validation12.csv')

    test = pd.read_csv('banksim/batches_test1clean.csv')

    return train, validation, test

def batchConstruction3(batch_size, cutoff_lenght, stage):
    print(batch_size)
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    #batch_size = 20
    aux_batch_size = batch_size
    #cutoff_lenght = 10
    withFraud = 0.25
    NoFraud = 1 - withFraud
    CusNoFraud = []
    CusFraud = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
    

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0
    n_fraud_cards = int(batch_size*withFraud)
    n_nofraud_cards = batch_size - n_fraud_cards
    while len(countFraudAppe)<N_Fraud and len(CusFraud) >= n_fraud_cards and (batches_with_fraud+1)/(batch_iteration_n+1) > 0.70 :
    #for i in range(2):
        #print('sim')
        batch_iteration_n = batch_iteration_n +1
        
        fraud_selected = []
        nofraud_selected = []
        b1 = False
        b2 = False
        
        while not b1:
            repeat = False
            fraud_selected = random.sample(CusFraud, n_fraud_cards)
            #print(fraud_selected)
            for x in fraud_selected:
                cardIndex = FraudIndex[x]
                if len(cardIndex) <= cutoff_lenght:
                    print('cartao ficou ppequeno fraude: ' + str(x))
                    CusFraud.remove(x)
                    repeat = True
                    break
                if x in countFraudAppe:
                    countFraudAppe[x] = countFraudAppe[x] + 1
                else:
                    countFraudAppe[x] = 1
            if repeat:
                continue
            b1 = True
        
        while not b2:
            repeat = False
            nofraud_selected = random.sample(CusNoFraud, n_nofraud_cards)
            for x in nofraud_selected:
                cardIndex = NoFraudIndex[x]
                if len(cardIndex) < cutoff_lenght:
                    print('cartao ficou ppequeno sem fraude ' + str(x))
                    CusNoFraud.remove(x)
                    repeat = True
                    break
            if repeat:
                continue
            b2 = True
        has_fraud = False
        verify_batch = 0
        while aux_batch_size > 0:
            #print(aux_batch_size)
            #acrescentar seed
            ran = random.uniform(0, 1)
            #caso random selecione um cartao com fraude
            if (ran > NoFraud and fraud_selected) or (not nofraud_selected and fraud_selected):
                cardName = fraud_selected.pop(0)
                cardIndex = FraudIndex[cardName]
                card_dir = 'Fraud/'
                #card with fraud
            #caso em que escolhe cartao sem fraude
            else:
                cardName = nofraud_selected.pop(0)
                cardIndex = NoFraudIndex[cardName]
                card_dir = 'NoFraud/'
                #card with no frauds
            aux_batch_size = aux_batch_size -1
            

            begin = cardIndex[0]
            # print(str(cardName) )
            # print( cardIndex)
            transactionIndex = cardIndex[cutoff_lenght-1]+1
            
            cus = pd.read_csv(dir + card_dir + cardName + '.csv')
            selected = cus.iloc[begin:transactionIndex]
            if len(selected) != cutoff_lenght:
                print('this selected is not with the right size', aux_batch_size)
                print(cardName)
                print(cardIndex)
                print(selected)
                print('begin', begin)
                print('end', transactionIndex)
                print(cus)
            cardIndex = cardIndex[cutoff_lenght:]
            if card_dir == 'Fraud/':
                for i in range(cutoff_lenght):
                    if 1 in selected['fraud']:
                        has_fraud = True

                FraudIndex[cardName] = cardIndex
            else:
                NoFraudIndex[cardName] = cardIndex
            verify_batch =  verify_batch + len(selected)
            
            
            #print('tamanho: ' + str(len(selected)))
            batch.append(selected)
            # for k in range(cutoff_lenght):
            #     batch.append(selected.iloc[k])
        if verify_batch != batch_size*cutoff_lenght:
            print("batch com erro " + str(batch_iteration_n))
            print('tamanho do batch', verify_batch)
            break
        if has_fraud:
            batches_with_fraud  = batches_with_fraud + 1
        else:
            print('batch with no frauds')
        aux_batch_size = batch_size
        aux_batch = pd.concat(batch, ignore_index= True)
        #print(aux_batch)
        batches.append(aux_batch)
        batch = []
        df_batches = pd.concat(batches, ignore_index = True) 
        
        print('batch completo n:' + str(batch_iteration_n))
        print(len(countFraudAppe))
        print(N_Fraud)
    #print(batches[0])
    #print(len(batches))
    print('batches with fraud: ' + str(batches_with_fraud))
    print('percentages batches with fraud: ' + str(batches_with_fraud/batch_iteration_n))
    return (df_batches, batch_iteration_n)

def batchConstruction4(batch_size, cutoff_lenght, stage): #Batches all have frauds
    print(batch_size)
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    aux_batch_size = batch_size
    avgbatches = 0
    if stage == 'Train':
            #avgbatches = 25000/(cutoff_lenght*batch_size)
            avgbatches = 2000
    elif stage == 'Validation':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    elif stage ==  'Test':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    #cutoff_lenght = 10
    withFraud = 0.5
    NoFraud = 1 - withFraud
    CusNoFraud = []
    CusFraud = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
    

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0
    n_fraud_cards = int(batch_size*withFraud)
    n_nofraud_cards = batch_size - n_fraud_cards
    while batch_iteration_n<avgbatches and len(countFraudAppe)<N_Fraud and len(CusFraud) >= n_fraud_cards:# and (batches_with_fraud+1)/(batch_iteration_n+1) > 0.70 :
    #for i in range(2):
        #print('sim')
        
        batch = []
        fraud_selected = []
        nofraud_selected = []
        b1 = False
        b2 = False
        
        while not b1:
            repeat = False
            fraud_selected = random.sample(CusFraud, n_fraud_cards)
            #print(fraud_selected)
            for x in fraud_selected:
                cardIndex = FraudIndex[x]
                if len(cardIndex) <= cutoff_lenght:
                    print('cartao ficou ppequeno fraude: ' + str(x))
                    CusFraud.remove(x)
                    repeat = True
                    break
                if x in countFraudAppe:
                    countFraudAppe[x] = countFraudAppe[x] + 1
                else:
                    countFraudAppe[x] = 1
            if repeat:
                continue
            b1 = True

        while not b2:
            repeat = False
            nofraud_selected = random.sample(CusNoFraud, n_nofraud_cards)
            for x in nofraud_selected:
                cardIndex = NoFraudIndex[x]
                if len(cardIndex) < cutoff_lenght:
                    print('cartao ficou ppequeno sem fraude ' + str(x))
                    CusNoFraud.remove(x)
                    repeat = True
                    break
            if repeat:
                continue
            b2 = True
        has_fraud = False
        verify_batch = 0
        while aux_batch_size > 0:
            #print(aux_batch_size)
            #acrescentar seed
            ran = random.uniform(0, 1)
            
            #caso random selecione um cartao com fraude
            if (ran > NoFraud and fraud_selected) or (not nofraud_selected and fraud_selected):
                cardName = fraud_selected.pop(0)
                cardIndex = FraudIndex[cardName]
                card_dir = 'Fraud/'
                #card with fraud
            #caso em que escolhe cartao sem fraude
            else:
                cardName = nofraud_selected.pop(0)
                cardIndex = NoFraudIndex[cardName]
                card_dir = 'NoFraud/'
                #card with no frauds
            aux_batch_size = aux_batch_size -1
            

            begin = cardIndex[0]
            # print(str(cardName) )
            # print( cardIndex)
            transactionIndex = cardIndex[cutoff_lenght-1]+1
            
            cus = pd.read_csv(dir + card_dir + cardName + '.csv')
            selected = cus.iloc[begin:transactionIndex]
            if len(selected) != cutoff_lenght:
                print('this selected is not with the right size', aux_batch_size)
                print(cardName)
                print(cardIndex)
                print(selected)
                print('begin', begin)
                print('end', transactionIndex)
                print(cus)
            cardIndex = cardIndex[1:]
            if card_dir == 'Fraud/':
                if selected['fraud'].iloc[-1] == 1:
                    has_fraud = True
                # if fraud_i != cutoff_lenght-1:
                    
                #     transactionIndex = transactionIndex - (cutoff_lenght-1 -i)
                #     begin = transactionIndex - (cutoff_lenght-1)
                #     cus = pd.read_csv(dir + card_dir + cardName + '.csv')
                #     selected = cus.iloc[begin:transactionIndex]
                
                FraudIndex[cardName] = cardIndex
            else:
                NoFraudIndex[cardName] = cardIndex
            verify_batch =  verify_batch + len(selected)
            
            
            #print('tamanho: ' + str(len(selected)))
            batch.append(selected)
            # for k in range(cutoff_lenght):
            #     batch.append(selected.iloc[k])
        if verify_batch != batch_size*cutoff_lenght:
            print("batch com erro " + str(batch_iteration_n))
            print('tamanho do batch', verify_batch)
            break
        aux_batch_size = batch_size
        if has_fraud:
            batches_with_fraud  = batches_with_fraud + 1
        else:
            print('batch with no frauds')
            continue
        batch_iteration_n = batch_iteration_n +1
        aux_batch = pd.concat(batch, ignore_index= True)
        #print(aux_batch)
        batches.append(aux_batch)
        
        df_batches = pd.concat(batches, ignore_index = True) 
        
        print('batch completo n:' + str(batch_iteration_n))
        print(len(countFraudAppe))
        print(N_Fraud)
    #print(batches[0])
    #print(len(batches))
    print('batches with fraud: ' + str(batches_with_fraud))
    print('percentages batches with fraud: ' + str(batches_with_fraud/batch_iteration_n))
    return (df_batches, batch_iteration_n)

def batchConstructionSmall(cutoff_lenght, stage, batch_size = 1): #Batches all have frauds
    print(batch_size)
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    aux_batch_size = batch_size
    avgbatches = 0
    if stage == 'Train':
            #avgbatches = 25000/(cutoff_lenght*batch_size)
            avgbatches = 1000
    elif stage == 'Validation':
            avgbatches = 50/(cutoff_lenght*batch_size)
    elif stage ==  'Test':
            avgbatches = 50/(cutoff_lenght*batch_size)
    #cutoff_lenght = 10
    withFraud = 0.5
    NoFraud = 1 - withFraud
    CusNoFraud = []
    CusFraud = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
    

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0
    # n_fraud_cards = int(batch_size*withFraud)
    # n_nofraud_cards = batch_size - n_fraud_cards
    n_fraud_cards = 1
    n_nofraud_cards = 1
    while batch_iteration_n<avgbatches:# and (batches_with_fraud+1)/(batch_iteration_n+1) > 0.70 :
    #for i in range(2):
        #print('sim')
        
        batch = []
        fraud_selected = []
        nofraud_selected = []
        b_rand = random.randint(0,2)
        if b_rand:
            b2 = True #doesnt go in the second while
            b1 = False
        else:
            b1 = True #doesnt go in the first while
            b2 = False
        # b1 = False
        # b2 = False
        
        while not b1:
            repeat = False
            fraud_selected = random.sample(CusFraud, n_fraud_cards)
            #print(fraud_selected)
            for x in fraud_selected:
                cardIndex = FraudIndex[x]
                if len(cardIndex) <= cutoff_lenght:
                    print('cartao ficou ppequeno fraude: ' + str(x))
                    CusFraud.remove(x)
                    repeat = True
                    break
                if x in countFraudAppe:
                    countFraudAppe[x] = countFraudAppe[x] + 1
                else:
                    countFraudAppe[x] = 1
            if repeat:
                continue
            b1 = True

        while not b2:
            repeat = False
            nofraud_selected = random.sample(CusNoFraud, n_nofraud_cards)
            for x in nofraud_selected:
                cardIndex = NoFraudIndex[x]
                if len(cardIndex) < cutoff_lenght:
                    print('cartao ficou ppequeno sem fraude ' + str(x))
                    CusNoFraud.remove(x)
                    repeat = True
                    break
            if repeat:
                continue
            b2 = True
        has_fraud = False
        verify_batch = 0
        
        #print(aux_batch_size)
        #acrescentar seed
        #ran = random.uniform(0, 1)
        

        #caso random selecione um cartao com fraude
        if not nofraud_selected and fraud_selected:
            cardName = fraud_selected.pop(0)
            cardIndex = FraudIndex[cardName]
            card_dir = 'Fraud/'
            #card with fraud
        #caso em que escolhe cartao sem fraude
        else:
            cardName = nofraud_selected.pop(0)
            cardIndex = NoFraudIndex[cardName]
            card_dir = 'NoFraud/'
            #card with no frauds
        aux_batch_size = aux_batch_size -1
        

        begin = cardIndex[0]
        # print(str(cardName) )
        # print( cardIndex)
        transactionIndex = cardIndex[cutoff_lenght-1]+1
        
        cus = pd.read_csv(dir + card_dir + cardName + '.csv')
        selected = cus.iloc[begin:transactionIndex]
        if len(selected) != cutoff_lenght:
            print('this selected is not with the right size', aux_batch_size)
            print(cardName)
            print(cardIndex)
            print(selected)
            print('begin', begin)
            print('end', transactionIndex)
            print(cus)
        cardIndex = cardIndex[1:]
        if card_dir == 'Fraud/':
            if selected['fraud'].iloc[-1] == 1:
                has_fraud = True
            # if fraud_i != cutoff_lenght-1:
                
            #     transactionIndex = transactionIndex - (cutoff_lenght-1 -i)
            #     begin = transactionIndex - (cutoff_lenght-1)
            #     cus = pd.read_csv(dir + card_dir + cardName + '.csv')
            #     selected = cus.iloc[begin:transactionIndex]
            
            FraudIndex[cardName] = cardIndex
        else:
            NoFraudIndex[cardName] = cardIndex
        # verify_batch =  verify_batch + len(selected)
        
        
        #print('tamanho: ' + str(len(selected)))
        batch.append(selected)
        # for k in range(cutoff_lenght):
        #     batch.append(selected.iloc[k])
        # if verify_batch != batch_size*cutoff_lenght:
        #     print("batch com erro " + str(batch_iteration_n))
        #     print('tamanho do batch', verify_batch)
        #     break
        aux_batch_size = batch_size
        if has_fraud:
            batches_with_fraud  = batches_with_fraud + 1
        # else:
        #     print('batch with no frauds')
        #     continue
        batch_iteration_n = batch_iteration_n +1
        aux_batch = pd.concat(batch, ignore_index= True)
        #print(aux_batch)
        batches.append(aux_batch)
        
        df_batches = pd.concat(batches, ignore_index = True) 
        
        print('batch completo n:' + str(batch_iteration_n))
        print(len(countFraudAppe))
        print(N_Fraud)
    #print(batches[0])
    #print(len(batches))
    print('batches with fraud: ' + str(batches_with_fraud))
    print('percentages batches with fraud: ' + str(batches_with_fraud/batch_iteration_n))
    return (df_batches, batch_iteration_n)

def batchConstruction1(cutoff_lenght, stage, batch_size = 1): #Batches all have frauds
    print(batch_size)
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    aux_batch_size = batch_size
    avgbatches = 0
    if stage == 'Train':
            #avgbatches = 25000/(cutoff_lenght*batch_size)
            avgbatches = 6000
    elif stage == 'Validation':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    elif stage ==  'Test':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    #cutoff_lenght = 10
    withFraud = 0.5
    NoFraud = 1 - withFraud
    CusNoFraud = []
    CusFraud = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
    

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0
    # n_fraud_cards = int(batch_size*withFraud)
    # n_nofraud_cards = batch_size - n_fraud_cards
    n_fraud_cards = 1
    n_nofraud_cards = 1
    while batch_iteration_n<avgbatches:# and (batches_with_fraud+1)/(batch_iteration_n+1) > 0.70 :
    #for i in range(2):
        #print('sim')
        
        batch = []
        fraud_selected = []
        nofraud_selected = []
        b_rand = random.randint(0,2)
        if b_rand:
            b2 = True #doesnt go in the second while
            b1 = False
        else:
            b1 = True #doesnt go in the first while
            b2 = False
        # b1 = False
        # b2 = False
        
        while not b1:
            repeat = False
            fraud_selected = random.sample(CusFraud, n_fraud_cards)
            #print(fraud_selected)
            for x in fraud_selected:
                cardIndex = FraudIndex[x]
                if len(cardIndex) <= cutoff_lenght:
                    print('cartao ficou ppequeno fraude: ' + str(x))
                    CusFraud.remove(x)
                    repeat = True
                    break
                if x in countFraudAppe:
                    countFraudAppe[x] = countFraudAppe[x] + 1
                else:
                    countFraudAppe[x] = 1
            if repeat:
                continue
            b1 = True

        while not b2:
            repeat = False
            nofraud_selected = random.sample(CusNoFraud, n_nofraud_cards)
            for x in nofraud_selected:
                cardIndex = NoFraudIndex[x]
                if len(cardIndex) < cutoff_lenght:
                    print('cartao ficou ppequeno sem fraude ' + str(x))
                    CusNoFraud.remove(x)
                    repeat = True
                    break
            if repeat:
                continue
            b2 = True
        has_fraud = False
        verify_batch = 0
        
        #print(aux_batch_size)
        #acrescentar seed
        #ran = random.uniform(0, 1)
        

        #caso random selecione um cartao com fraude
        if not nofraud_selected and fraud_selected:
            cardName = fraud_selected.pop(0)
            cardIndex = FraudIndex[cardName]
            card_dir = 'Fraud/'
            #card with fraud
        #caso em que escolhe cartao sem fraude
        else:
            cardName = nofraud_selected.pop(0)
            cardIndex = NoFraudIndex[cardName]
            card_dir = 'NoFraud/'
            #card with no frauds
        aux_batch_size = aux_batch_size -1
        

        begin = cardIndex[0]
        # print(str(cardName) )
        # print( cardIndex)
        transactionIndex = cardIndex[cutoff_lenght-1]+1
        
        cus = pd.read_csv(dir + card_dir + cardName + '.csv')
        selected = cus.iloc[begin:transactionIndex]
        if len(selected) != cutoff_lenght:
            print('this selected is not with the right size', aux_batch_size)
            print(cardName)
            print(cardIndex)
            print(selected)
            print('begin', begin)
            print('end', transactionIndex)
            print(cus)
        cardIndex = cardIndex[1:]
        if card_dir == 'Fraud/':
            if selected['fraud'].iloc[-1] == 1:
                has_fraud = True
            # if fraud_i != cutoff_lenght-1:
                
            #     transactionIndex = transactionIndex - (cutoff_lenght-1 -i)
            #     begin = transactionIndex - (cutoff_lenght-1)
            #     cus = pd.read_csv(dir + card_dir + cardName + '.csv')
            #     selected = cus.iloc[begin:transactionIndex]
            
            FraudIndex[cardName] = cardIndex
        else:
            NoFraudIndex[cardName] = cardIndex
        # verify_batch =  verify_batch + len(selected)
        
        
        #print('tamanho: ' + str(len(selected)))
        batch.append(selected)
        # for k in range(cutoff_lenght):
        #     batch.append(selected.iloc[k])
        # if verify_batch != batch_size*cutoff_lenght:
        #     print("batch com erro " + str(batch_iteration_n))
        #     print('tamanho do batch', verify_batch)
        #     break
        aux_batch_size = batch_size
        if has_fraud:
            batches_with_fraud  = batches_with_fraud + 1
        # else:
        #     print('batch with no frauds')
        #     continue
        batch_iteration_n = batch_iteration_n +1
        aux_batch = pd.concat(batch, ignore_index= True)
        #print(aux_batch)
        batches.append(aux_batch)
        
        df_batches = pd.concat(batches, ignore_index = True) 
        
        print('batch completo n:' + str(batch_iteration_n))
        print(len(countFraudAppe))
        print(N_Fraud)
    #print(batches[0])
    #print(len(batches))
    print('batches with fraud: ' + str(batches_with_fraud))
    print('percentages batches with fraud: ' + str(batches_with_fraud/batch_iteration_n))
    return (df_batches, batch_iteration_n)

def batchConstructionTest(batch_size, cutoff_lenght, stage): #Batches all have frauds
    print(batch_size)
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    aux_batch_size = batch_size
    avgbatches = 0
    if stage == 'Train':
            #avgbatches = 25000/(cutoff_lenght*batch_size)
            avgbatches = 1000
    elif stage == 'Validation':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    elif stage ==  'Test':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    #cutoff_lenght = 10
    withFraud = 0.5
    NoFraud = 1 - withFraud
    CusNoFraud = []
    CusFraud = []
    Cus = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
        Cus.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
        Cus.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
        Cus.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
        Cus.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
    

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0
    
    while batch_iteration_n<avgbatches :# and (batches_with_fraud+1)/(batch_iteration_n+1) > 0.70 :
    #for i in range(2):
        #print('sim')
        
        batch = []
        fraud_selected = []
        nofraud_selected = []
        selected = []
        b1 = False
        b2 = False
        b3 = False
        while not b3:
            repeat = False
            just_selected = random.sample(Cus, batch_size)
            #print(fraud_selected)
            for x in just_selected:
                if x in CusFraud:
                    cardIndex = FraudIndex[x]
                    if len(cardIndex) <= cutoff_lenght:
                        print('cartao ficou ppequeno fraude: ' + str(x))
                        CusFraud.remove(x)
                        Cus.remove(x)
                        repeat = True
                        break
                    if x in countFraudAppe:
                        countFraudAppe[x] = countFraudAppe[x] + 1
                    else:
                        countFraudAppe[x] = 1
                elif x in CusNoFraud:
                    cardIndex = NoFraudIndex[x]
                    if len(cardIndex) < cutoff_lenght:
                        print('cartao ficou ppequeno sem fraude ' + str(x))
                        CusNoFraud.remove(x)
                        Cus.remove(x)
                        repeat = True
                        break
            if repeat:
                continue
            b3 = True
        has_fraud = False
        verify_batch = 0
        while aux_batch_size > 0:
            #print(aux_batch_size)
            #acrescentar seed
            ran = random.uniform(0, 1)
            
            if just_selected:
                cardName = just_selected.pop(0)
                if cardName in CusFraud:
                    cardIndex = FraudIndex[cardName]
                    card_dir = 'Fraud/'
                elif cardName in CusNoFraud:
                    cardIndex = NoFraudIndex[cardName]
                    card_dir = 'NoFraud/'

            aux_batch_size = aux_batch_size -1
            

            begin = cardIndex[0]
            # print(str(cardName) )
            # print( cardIndex)
            transactionIndex = cardIndex[cutoff_lenght-1]+1
            
            cus = pd.read_csv(dir + card_dir + cardName + '.csv')
            selected = cus.iloc[begin:transactionIndex]
            if len(selected) != cutoff_lenght:
                print('this selected is not with the right size', aux_batch_size)
                print(cardName)
                print(cardIndex)
                print(selected)
                print('begin', begin)
                print('end', transactionIndex)
                print(cus)
            cardIndex = cardIndex[1:]
            if card_dir == 'Fraud/':
                if selected['fraud'].iloc[-1] == 1:
                    has_fraud = True
                # if fraud_i != cutoff_lenght-1:
                    
                #     transactionIndex = transactionIndex - (cutoff_lenght-1 -i)
                #     begin = transactionIndex - (cutoff_lenght-1)
                #     cus = pd.read_csv(dir + card_dir + cardName + '.csv')
                #     selected = cus.iloc[begin:transactionIndex]
                
                FraudIndex[cardName] = cardIndex
            else:
                NoFraudIndex[cardName] = cardIndex
            verify_batch =  verify_batch + len(selected)
            
            
            #print('tamanho: ' + str(len(selected)))
            batch.append(selected)
            # for k in range(cutoff_lenght):
            #     batch.append(selected.iloc[k])
        if verify_batch != batch_size*cutoff_lenght:
            print("batch com erro " + str(batch_iteration_n))
            print('tamanho do batch', verify_batch)
            break
        aux_batch_size = batch_size
        if has_fraud:
            batches_with_fraud  = batches_with_fraud + 1
        # else:
        #     #print('batch with no frauds')
        #     continue
        batch_iteration_n = batch_iteration_n +1
        aux_batch = pd.concat(batch, ignore_index= True)
        #print(aux_batch)
        batches.append(aux_batch)
        
        df_batches = pd.concat(batches, ignore_index = True) 
        
        print('batch completo n:' + str(batch_iteration_n))
        print(len(countFraudAppe))
        print(N_Fraud)
    #print(batches[0])
    #print(len(batches))
    print('batches with fraud: ' + str(batches_with_fraud))
    print('percentages batches with fraud: ' + str(batches_with_fraud/batch_iteration_n))
    return (df_batches, batch_iteration_n)

def batchConstruction1Test(cutoff_lenght, stage, batch_size = 1): #Batches all have frauds
    print(batch_size)
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    aux_batch_size = batch_size
    avgbatches = 0
    if stage == 'Train':
            #avgbatches = 25000/(cutoff_lenght*batch_size)
            avgbatches = 10000
    elif stage == 'Validation':
            avgbatches = 5000/(cutoff_lenght*batch_size)
    elif stage ==  'Test':
            avgbatches = 15000/(cutoff_lenght*batch_size)
    #cutoff_lenght = 10
    withFraud = 0.5
    NoFraud = 1 - withFraud
    CusNoFraud = []
    CusFraud = []
    Cus = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
        Cus.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
        Cus.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
        Cus.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<cutoff_lenght:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
        Cus.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
    

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0
    # n_fraud_cards = int(batch_size*withFraud)
    # n_nofraud_cards = batch_size - n_fraud_cards
    n_fraud_cards = 1
    n_nofraud_cards = 1
    while batch_iteration_n<avgbatches:# and (batches_with_fraud+1)/(batch_iteration_n+1) > 0.70 :
    #for i in range(2):
        #print('sim')
        
        batch = []
        fraud_selected = []
        nofraud_selected = []
        select = []
        b_rand = random.randint(0,2)
        if b_rand:
            b2 = True #doesnt go in the second while
            b1 = False
        else:
            b1 = True #doesnt go in the first while
            b2 = False
        # b1 = False
        # b2 = False
        b3 = False
        while not b3:
            repeat = False
            just_selected = random.sample(Cus, batch_size)
            #print(fraud_selected)
            for x in just_selected:
                if x in CusFraud:
                    cardIndex = FraudIndex[x]
                    if len(cardIndex) <= cutoff_lenght:
                        print('cartao ficou ppequeno fraude: ' + str(x))
                        CusFraud.remove(x)
                        Cus.remove(x)
                        repeat = True
                        break
                    if x in countFraudAppe:
                        countFraudAppe[x] = countFraudAppe[x] + 1
                    else:
                        countFraudAppe[x] = 1
                elif x in CusNoFraud:
                    cardIndex = NoFraudIndex[x]
                    if len(cardIndex) < cutoff_lenght:
                        print('cartao ficou ppequeno sem fraude ' + str(x))
                        CusNoFraud.remove(x)
                        Cus.remove(x)
                        repeat = True
                        break
            if repeat:
                continue
            b3 = True
       
        has_fraud = False
        verify_batch = 0
        
        #print(aux_batch_size)
        #acrescentar seed
        #ran = random.uniform(0, 1)
        

        #caso random selecione um cartao com fraude
        if just_selected:
            if just_selected:
                cardName = just_selected.pop(0)
                if cardName in CusFraud:
                    cardIndex = FraudIndex[cardName]
                    card_dir = 'Fraud/'
                elif cardName in CusNoFraud:
                    cardIndex = NoFraudIndex[cardName]
                    card_dir = 'NoFraud/'
            #card with fraud
        #caso em que escolhe cartao sem fraude
        aux_batch_size = aux_batch_size -1
        

        begin = cardIndex[0]
        # print(str(cardName) )
        # print( cardIndex)
        transactionIndex = cardIndex[cutoff_lenght-1]+1
        
        cus = pd.read_csv(dir + card_dir + cardName + '.csv')
        selected = cus.iloc[begin:transactionIndex]
        if len(selected) != cutoff_lenght:
            print('this selected is not with the right size', aux_batch_size)
            print(cardName)
            print(cardIndex)
            print(selected)
            print('begin', begin)
            print('end', transactionIndex)
            print(cus)
        cardIndex = cardIndex[1:]
        if card_dir == 'Fraud/':
            if selected['fraud'].iloc[-1] == 1:
                has_fraud = True
            # if fraud_i != cutoff_lenght-1:
                
            #     transactionIndex = transactionIndex - (cutoff_lenght-1 -i)
            #     begin = transactionIndex - (cutoff_lenght-1)
            #     cus = pd.read_csv(dir + card_dir + cardName + '.csv')
            #     selected = cus.iloc[begin:transactionIndex]
            
            FraudIndex[cardName] = cardIndex
        else:
            NoFraudIndex[cardName] = cardIndex
        # verify_batch =  verify_batch + len(selected)
        
        
        #print('tamanho: ' + str(len(selected)))
        batch.append(selected)
        # for k in range(cutoff_lenght):
        #     batch.append(selected.iloc[k])
        # if verify_batch != batch_size*cutoff_lenght:
        #     print("batch com erro " + str(batch_iteration_n))
        #     print('tamanho do batch', verify_batch)
        #     break
        aux_batch_size = batch_size
        if has_fraud:
            batches_with_fraud  = batches_with_fraud + 1
        # else:
        #     print('batch with no frauds')
        #     continue
        batch_iteration_n = batch_iteration_n +1
        aux_batch = pd.concat(batch, ignore_index= True)
        #print(aux_batch)
        batches.append(aux_batch)
        
        df_batches = pd.concat(batches, ignore_index = True) 
        
        print('batch completo n:' + str(batch_iteration_n))
        print(len(countFraudAppe))
        print(N_Fraud)
    #print(batches[0])
    #print(len(batches))
    print('batches with fraud: ' + str(batches_with_fraud))
    print('percentages batches with fraud: ' + str(batches_with_fraud/batch_iteration_n))
    return (df_batches, batch_iteration_n)

def datasetWorkTogether(stage):
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return

    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    CusNoFraud = []
    CusFraud = []

    #df_complete = pd.Dataframe()

    frames = []

    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        frames = frames + [cus.copy()]
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        frames = frames + [cus.copy()]

    Data = pd.concat(frames)
    print(Data.columns)
    Data = Data.sort_values(by='Unnamed: 0')
    print(Data.head())

    Data.to_csv('banksim/DatasetWork/bankSimDataWorked' + stage + '.csv', index =False)


def batchConstruction5(cutoff_lenght,stage):
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir_original = 'banksim/DatasetWork/bankSimDataWorked'+ stage + '.csv'

    data = pd.read_csv(dir_original)

    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    #cutoff_lenght = 10
    withFraud = 0.5
    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0

    total_transactions = len(data.index)
    trans_i = 0
    customers = data['customer']


    DataPerApp = {}
    LenPerApp = {}
    batch_iteration_n = 0
    while trans_i < total_transactions:
        cus = customers.iloc[trans_i]
        if cus in LenPerApp:
            LenPerApp[cus] = LenPerApp[cus] + 1
            transaction = data.iloc[[trans_i]]
            #print("this shit")
            #print(DataPerApp[cus])
            DataPerApp[cus] =  DataPerApp[cus].append(transaction)
            #print(DataPerApp[cus])
            if LenPerApp[cus] == cutoff_lenght:
                #print(DataPerApp[cus])
                selected = DataPerApp[cus]
                # print(type(selected))
                # print(LenPerApp[cus])
                #print(aux_batch)
                batches.append(selected.copy())
                
                LenPerApp[cus] = 0
                DataPerApp[cus] = pd.DataFrame()
                batch_iteration_n = batch_iteration_n +1
                print('It is batch number:' + str(batch_iteration_n))
        else:
            LenPerApp[cus] = 1
            DataPerApp[cus] = data.iloc[[trans_i]]
        # if trans_i == 100:
        #             print(DataPerApp)
        trans_i = trans_i+1

    df_batches = pd.concat(batches, ignore_index = True)
    
    return (df_batches, batch_iteration_n)


def batchConstruction6(cutoff_lenght,stage):
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir_original = 'banksim/DatasetWork/bankSimDataWorked'+ stage + '.csv'

    data = pd.read_csv(dir_original)

    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    #cutoff_lenght = 10
    withFraud = 0.5
    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes

#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0

    total_transactions = len(data.index)
    trans_i = 0
    customers = data['customer']


    DataPerApp = {}
    LenPerApp = {}
    batch_iteration_n = 0
    while trans_i < total_transactions:
        cus = customers.iloc[trans_i]
        if cus in LenPerApp:
            LenPerApp[cus] = LenPerApp[cus] + 1
            transaction = data.iloc[[trans_i]]
            #print("this shit")
            #print(DataPerApp[cus])
            DataPerApp[cus] =  DataPerApp[cus].append(transaction)
            #print(DataPerApp[cus])
            if LenPerApp[cus] == cutoff_lenght:
                #print(DataPerApp[cus])
                selected = DataPerApp[cus]
                # print(type(selected))
                # print(LenPerApp[cus])
                #print(aux_batch)
                batches.append(selected.copy())
                
                LenPerApp[cus] = LenPerApp[cus] - 1
                DataPerApp[cus] = DataPerApp[cus].iloc[1:]
                #print(DataPerApp[cus])
                batch_iteration_n = batch_iteration_n +1
                print('It is batch number:' + str(batch_iteration_n))
        else:
            LenPerApp[cus] = 1
            DataPerApp[cus] = data.iloc[[trans_i]]
        # if trans_i == 100:
        #             print(DataPerApp)
        trans_i = trans_i+1

    df_batches = pd.concat(batches, ignore_index = True)
    
    return (df_batches, batch_iteration_n)


def batchConstruction7(cutoff_lenght,stage):
    if stage != 'Train' and stage != 'Validation' and stage != 'Test':
        print('Stage has an invalid value')
        return
    dir_original = 'banksim/DatasetWork/bankSimDataWorked'+ stage + '.csv'
    batch_size = -1
    if stage == 'Train':
            #avgbatches = 25000/(cutoff_lenght*batch_size)
            batches = 10
    elif stage == 'Validation':
            batches = 3
    elif stage ==  'Test':
            batches = 3
    min_n_trans = cutoff_lenght*batch_size

    data = pd.read_csv(dir_original)
    
    dir = 'banksim/DatasetWork/Divided/' + stage + '/'
    #cutoff_lenght = 10
    withFraud = 0.5
    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    CusNoFraud = []
    CusFraud = []
    NoFraudIndex = {}
    FraudIndex = {}
    f = open(dir + 'Fraud/CustomersWithFraud.txt', 'r')
    Lines = f.read().splitlines()
    for line in Lines:
        CusFraud.append(line)
    print(CusFraud[0])
    g = open(dir + 'NoFraud/CustomersNoFraud.txt', 'r')
    Lines2 = g.read().splitlines()
    for line in Lines2:
        CusNoFraud.append(line)
    print('---------------')
    print(CusNoFraud[0])

    #randomly choose 75% of cards with no fraud and 25% of cards with fraudCus
    #first build the idexes
    to_remove = []
    for x in CusNoFraud:
        cus = pd.read_csv(dir + 'NoFraud/' + x + '.csv')
        N = len(cus)
        if N<min_n_trans:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        NoFraudIndex[x] = aux
    for u in to_remove:
        CusNoFraud.remove(u)
    to_remove = []
    for x in CusFraud:
        cus = pd.read_csv(dir + 'Fraud/' + x + '.csv')
        N = len(cus)
        if N<min_n_trans:
            to_remove = to_remove + [x]
            continue
        aux = np.arange(0, N, 1)
        FraudIndex[x] = aux
    for u in to_remove:
        CusFraud.remove(u)
    
    N_NoFraud = len(NoFraudIndex)
    N_Fraud = len(FraudIndex)
#falta ver tipo certo para dar output mas por agora vai em lista
    batch = []
    batches = []
    #df_batch = 
    text = ''
    batch_iteration_n = 0
    countFraudAppe = {}
    batches_with_fraud = 0

    total_transactions = len(data.index)
    trans_i = 0
    customers = data['customer']


    DataPerApp = {}
    LenPerApp = {}
    batch_iteration_n = 0
    while trans_i < total_transactions:
        cus = customers.iloc[trans_i]
        if cus in LenPerApp:
            LenPerApp[cus] = LenPerApp[cus] + 1
            transaction = data.iloc[[trans_i]]
            #print("this shit")
            #print(DataPerApp[cus])
            DataPerApp[cus] =  DataPerApp[cus].append(transaction)
            #print(DataPerApp[cus])
            if LenPerApp[cus] == cutoff_lenght:
                #print(DataPerApp[cus])
                selected = DataPerApp[cus]
                # print(type(selected))
                # print(LenPerApp[cus])
                #print(aux_batch)
                batches.append(selected.copy())
                
                LenPerApp[cus] = LenPerApp[cus] - 1
                DataPerApp[cus] = DataPerApp[cus].iloc[1:]
                #print(DataPerApp[cus])
                batch_iteration_n = batch_iteration_n +1
                print('It is batch number:' + str(batch_iteration_n))
        else:
            LenPerApp[cus] = 1
            DataPerApp[cus] = data.iloc[[trans_i]]
        # if trans_i == 100:
        #             print(DataPerApp)
        trans_i = trans_i+1

    df_batches = pd.concat(batches, ignore_index = True)
    
    return (df_batches, batch_iteration_n)

#datasetWorkTogether('Train')
# buildBatches(12,2)
# buildTest(2)
#buildBatchesSeq(5)
#batchConstruction5(5,'Train')
# train, n_batches4 = batchConstruction5(5, 'Train')
# train.to_csv('banksim/batches5Seq.csv', index =False)