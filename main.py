import pandas as pd
import numpy as np

training_set = pd.read_csv('TRAINING.csv')
test_set = pd.read_csv('TEST.csv')

test_set.fillna(0, inplace = True)

id = training_set['id'] 
Area = training_set['Area(total)']
Troom = training_set['Troom']
Nbedrooms = training_set['Nbedrooms']
Nbwashrooms = training_set['Nbwashrooms']
Twashrooms = training_set['Twashrooms']
Roof = training_set['Roof(Area)']
Lawn = training_set['Lawn(Area)']
Nfloors = training_set['Nfloors']
Api = training_set['API']
Anb = training_set['ANB']
Grade = training_set['Grade']
#Price = training_set['Price']

idt = test_set['id'] 
Areat = test_set['Area(total)']
Troomt = test_set['Troom']
Nbedroomst = test_set['Nbedrooms']
Nbwashroomst = test_set['Nbwashrooms']
Twashroomst = test_set['Twashrooms']
Rooft = test_set['Roof(Area)']
Lawnt = test_set['Lawn(Area)']
Nfloorst = test_set['Nfloors']
Apit = test_set['API']
Anbt = test_set['ANB']
#Pricet = test_set['Price']



# Cleaning the training_set
# Removing the nan value and replacing it by 0

new_training_set = {}
new_test_set = {}


roof = training_set['roof']
r = []

for i in roof:
    if i == 0:
        r.append("NO")
    else:
        i = i.upper()
        r.append(i)
   
rooft = test_set['roof']
rt = []

for i in rooft:
    if i == 0:
        rt.append("NO")
    else:
        i = i.upper()
        rt.append(i)
        
        
prices = training_set['Price']
new_prices = []
for i in prices:
    new_prices.append(int(i[0:4]))
    
    
pricest = test_set['Price']
new_prices_test = []
for i in prices:
    new_prices_test.append(int(i[0:4]))
        


new_training_set['id'] = id
#new_training_set['Area'] = Area
new_training_set['Nbedrooms'] = Nbedrooms
new_training_set['Twashrooms'] = Twashrooms
new_training_set['roof'] = r
#new_training_set['Roof'] = Roof
new_training_set['Lawn'] = Lawn
new_training_set['Nfloors'] = Nfloors
new_training_set['Api'] = Api
new_training_set['Anb'] = Anb
#new_training_set['Price'] = new_prices
new_training_set['Grade'] = Grade

#new_test_set['Area'] = Areat
new_test_set['Nbedrooms'] = Nbedroomst
new_test_set['Twashrooms'] = Twashroomst
new_test_set['roof'] = rt
new_test_set['Roof'] = Rooft
new_test_set['Lawn'] = Lawnt
new_test_set['Nfloors'] = Nfloorst
new_test_set['Api'] = Apit
new_test_set['Anb'] = Anbt
#new_test_set['Price'] = pd.Series(new_prices_test)

new_test_set = pd.DataFrame(new_test_set)
new_training_set = pd.DataFrame(new_training_set)

#encoding the roof attribute
from sklearn.preprocessing import LabelEncoder

le_roof = LabelEncoder()
le_grade = LabelEncoder()
le_rooft = LabelEncoder()
le_roof.fit(r)
le_rooft.fit(rt)
le_grade.fit(Grade)
new_training_set['Grade'] = le_grade.transform(new_training_set['Grade'])
new_training_set['roof'] = le_roof.transform(new_training_set['roof'])
new_test_set['roof'] = le_rooft.transform(new_test_set['roof'])


# Dependent Variable
Y_train = new_training_set.iloc[:, -1].values  

# Independent Variables
X_train = new_training_set.iloc[:,1 : -1].values

# Independent Variables for Test Set
X_test = new_test_set.iloc[:,:].values

#feature scalling it 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
Y_train = Y_train.reshape((len(Y_train), 1)) 

Y_train = sc.fit_transform(Y_train)
Y_train = Y_train.ravel()

from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor( loss = 'huber', learning_rate=0.001, n_estimators=350, max_depth=6
                              ,subsample=1, verbose=False, random_state=126)

gbr.fit(X_train,Y_train)

y_pred_gbr = sc.inverse_transform(gbr.predict(X_test))
y_pred_gbr = pd.DataFrame(y_pred_gbr, columns = ['Grade']) # Converting to dataframe

        


