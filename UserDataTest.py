#This file allows the user to input their own data for each of the 16 client features.
#The previously trained model is then used to make a prediction on if the client will make a term deposit

import torch
from BankModel import TabularBankModel
from DataAnalysis import embedding_sizes, continuous_data
OUTPUT_SIZE = 2
LAYERS = [200,100]

model2 = TabularBankModel(embedding_sizes, continuous_data.shape[1], OUTPUT_SIZE, LAYERS, p=0.4)

#Load saved model and set up for evaluation of a single example
model2.load_state_dict(torch.load(r"C:\Users\VijayBehal\TabularBankModel.pt"))
model2.eval()

def test_data(model):
    #User inputs categorical data
    job = int(input("What is the client's job? Type the corresponding number:" +
             "\n0: Admin" +
             "\n1: Blue-collar" +
             "\n2: Entrepreneur" +
             "\n3: Housemaid" +
             "\n4: Management" +
             "\n5: Retired" +
             "\n6: Self-employed" +
             "\n7: Services" +
             "\n8: Student" +
             "\n9: Technician" +
             "\n10: Unemployed" +
             "\n11: Unknown\n"))
    
    marital = int(input("What is the client's marital status? Type the corresponding number:" +
             "\n0: Divorced or Widowed" +
             "\n1: Married" +
             "\n2: Single\n"))
    
    edu = int(input("What is the client's level of education? Type the corresponding number:" +
             "\n0: Primary" +
             "\n1: Secondary" +
             "\n2: Tertiary" +
             "\n3: Unknown\n"))
    
    credit = int(input("Does the client have credit in default? Type the corresponding number:" +
             "\n0: No" +
             "\n1: Yes\n"))
    
    house = int(input("Does the client have a housing loan? Type the corresponding number:" +
             "\n0: No" +
             "\n1: Yes\n"))
    
    personal = int(input("Does the client have a personal loan? Type the corresponding number:" +
             "\n0: No" +
             "\n1: Yes\n"))
    
    comm = int(input("How is the client being contacted? Type the corresponding number:" +
             "\n0: Cellular" +
             "\n1: Telephone" +
             "\n2: Unknown\n"))
    
    month = int(input("What month was the client last contacted? Type the corresponding number:" +
             "\n0: April" +
             "\n1: August" +
             "\n2: December" +
             "\n3: February" +
             "\n4: January" +
             "\n5: July" +
             "\n6: June" +
             "\n7: March" +
             "\n8: May" +
             "\n9: November" +
             "\n10: October" +
             "\n11: September\n"))
    
    prev_camp = int(input("What was the result of this client being contacted in a previous campaign? Type the corresponding number:" +
             "\n0: Failure" +
             "\n1: Other" +
             "\n2: Success" +
             "\n3: Unknown\n"))
    
    #Compile categorical data into tensor
    categorical = torch.tensor([job, marital, edu, credit, house, personal, comm, month, prev_camp], dtype=torch.int64).reshape(1,-1)

    #User inputs continuous data
    balance = float(input("What is the average yearly balance of the client, in euros? "))

    day = 0
    while day<1 or day>31:
        day = int(input("What day of the month was the client last contacted? "))
        if day<1 or day>31:
            print("Please enter a valid date")
    
    secs = float(input("What was the duration of the last contact, in seconds? "))

    times_contacted = int(input("How many times has the client been contacted during this campaign? "))

    days_since = int(input("How many days have passed since the client was last contacted? Enter '-1' if the client was not previously contacted: "))

    prev_contacts = int(input("How many times was the client contacted before this campaign? "))

    #Compile continuous data into tensor
    continuous = torch.tensor([balance, day, secs, times_contacted, days_since, prev_contacts], dtype=torch.float).reshape(1,-1)

    #Pass data through model
    with torch.no_grad():
        y_value = model(categorical, continuous)

    if y_value.argmax() == 0:
        print("The model estimates: No. This client will not subscribe to a term deposit")
    elif y_value.argmax() == 1:
        print("The model estimates: Yes. This client will subscribe to a term deposit")

if __name__ == '__main__':
    test_data(model2)
