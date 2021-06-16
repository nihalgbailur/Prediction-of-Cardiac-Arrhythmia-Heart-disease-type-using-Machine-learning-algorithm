import Tkinter #importing tkinter library for GUI creation
from Tkinter import *
from PIL import Image, ImageTk
import tkMessageBox
import tkFileDialog
import mysql.connector

import pandas as pnd  # importing pandas data analysis toolkit
import numpy as np    # importing numpy library for array operations
from time import time # importing time library for time calculations
from sklearn.model_selection import train_test_split # importing module model_classification from scikit-learn library

   
header_row = ['age', 'sex', 'pain', 'BP', 'chol', 'fbs', 'ecg', 'maxhr','eiang', 'eist', 'slope', 'vessels', 'thal',
              'diagnosis']     # Declaring the header row for getting data from the dataset files


# filter to only those diagnosed with heart disease
def cardiac():
    global master
    master = Tk()         # Defining the Tkinter widget
    master.wm_title("Heart Disease Prediction")
    master.geometry('1500x600')
    image = Image.open('c3.jpg')
    image = image.resize((1500, 600))
    photo_image = ImageTk.PhotoImage(image)
    label = Label(master, image = photo_image)
    label.place(x=0,y=0)
   
    import sklearn        # Importing scikit-learn functions

    #Lab=Label(master,text=" Automatic Heart Disease Detection ")    # Adding Label to the Tkinter widget
    #Lab.place(x=600,y=50)                           # Packing the label data to the tkinter widget in user defined rows and columns
                                         # Changing dimensions of the Label

    #Lab=Label(master,text="")
    #Lab.grid(row=2,column=5,columnspan=2)

    #Lab1=Label(master,text="Classification Report")
    #Lab1.place(x=170,y=330)
    #Lab2=Label(master,text="Confusion Matrix")
    #Lab2.place(x=930,y=330)    
    T = Text(master, height=6, width=40,font=("bold",10),highlightthickness=2,bg="white",relief=SUNKEN)                            # Declaring Text Widget for Result Displaying
    T.place(x=55,y=350)
    T1 = Text(master, height=6, width=35,font=("bold",10),highlightthickness=2,bg="white",relief=SUNKEN)
    T1.place(x=1000,y=350)

    var = StringVar(master)
    var.set("Select Dataset") # initial value

    option = OptionMenu(master, var, "Cleveland", "Hungarian", "VA", "all") # Declaring the OptionMenu (Drop-Down list) widget
    option.config(bg = "violet")
    option.config(fg = "black")
    option.config(font=('algerian',10,'bold'))
    option.config(width=12)
    option.place(x=500,y=80)

    '''field1="Age"                                                            # Defining the field names which user has to input for heart disease detection
    field2="Sex"
    field3="Pain"
    field4="BP"
    field5="Chol"
    field6="FBS"
    field7="ECG"
    field8="Maxhr"
    field9="Eiang"
    field10="Eist"
    field11="Slope" 
    field12="Vessels"
    field13="Thal"'''


    '''L1=Label(master,text=field1)
    L1.grid(row = 4, column = 0, sticky='nsew')
    L1.configure(width=14)
    L2=Label(master,text=field2)
    L2.grid(row = 4, column = 1, sticky='nsew')
    L2.configure(width=14)
    L3=Label(master,text=field3)
    L3.grid(row = 4, column = 2, sticky='nsew')
    L3.configure(width=14)
    L4=Label(master,text=field4)
    L4.grid(row = 4, column = 3, sticky='nsew')
    L4.configure(width=14)
    L5=Label(master,text=field5)
    L5.grid(row = 4, column = 4, sticky='nsew')
    L5.configure(width=14)
    L6=Label(master,text=field6, )
    L6.grid(row = 4, column = 5, sticky='nsew')
    L6.configure(width=14)
    L7=Label(master,text=field7)
    L7.grid(row = 4, column = 6, sticky='nsew')
    L7.configure(width=14)
    L8=Label(master,text=field8)
    L8.grid(row = 4, column = 7, sticky='nsew')
    L8.configure(width=14)
    L9=Label(master,text=field9)
    L9.grid(row = 4, column = 8, sticky='nsew')
    L9.configure(width=14)
    L10=Label(master,text=field10)
    L10.grid(row = 4, column = 9, sticky='nsew')
    L10.configure(width=14)
    L11=Label(master,text=field11)
    L11.grid(row = 4, column = 10, sticky='nsew')
    L11.configure(width=14)
    L12=Label(master,text=field12)
    L12.grid(row = 4, column = 11, sticky='nsew')
    L12.configure(width=14)
    L13=Label(master,text=field13)
    L13.grid(row = 4, column = 12, sticky='nsew')
    L13.configure(width=14)'''


    E1=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E1.place(x=0, y=220)

    E2=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E2.place(x=90, y=220)

    E3=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E3.place(x=180, y=220)

    E4=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E4.place(x=280, y=220)

    E5=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E5.place(x=370, y=220)

    E6=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E6.place(x=470, y=220)

    E7=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E7.place(x=550, y=220)

    E8=Entry(master,width=8,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E8.place(x=650, y=220)

    E9=Entry(master,width=10,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E9.place(x=770, y=220)

    E10=Entry(master,width=10,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E10.place(x=880, y=220)

    E11=Entry(master,width=10,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E11.place(x=980, y=220)

    E12=Entry(master,width=10,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E12.place(x=1100, y=220)

    E13=Entry(master,width=10,font=("bold",10),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E13.place(x=1220, y=220)

    lb1 = Label(master, text="patient",font=('algerian',15,'bold'),fg="BLACK",anchor='w')
    lb1.place(x=0, y=150)

    E0=Entry(master,width=10,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    E0.place(x=120, y=150)
    
   


    '''Labx=Label(master,text="")
    Labx.grid(row=21,column=4,columnspan=4)
    Labx.visible=False'''

    #T3 = Text(master, height=2, width=30)                                                       # Declaring Text Widget for Displaying Prediction
    #T3.grid(row=23,column=4, columnspan=4, sticky= 'nsew')

    def train_classifier(x_train,x_test,y_train,y_test,string):                         # Declaring the function for training classifiers and classification analysis
        global clf                                                                      # Declaring clf as a Global Variable for using throughot the code
        global outclass
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        T.delete(1.0,END)                                                               # Deleting the text in the Text Widget
        T1.delete(1.0,END)
        if string=="SVM":
            from sklearn import svm
            from sklearn.svm import SVC
            from sklearn.model_selection import GridSearchCV
        
            t1= time()
            param_grid = {'C': [1, 5, 10, 50, 100],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
            clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)  # Initialization of GridSearch Optimization for SVM with RBF kernel

            clf.fit(x_train, y_train)                                                   # Fitting the classifier to the training and testing the SVM Classifier    
            y_pred = clf.predict(x_test)
            print(y_pred)
            # Predict Results for Test Data
            title = "Learning Curves (SVM)"
            geterror(x_train,y_train,clf,title);
            t= time()-t1
            print("Training Complete")
        
        elif string=="Naive Bayes":
            from sklearn.naive_bayes import GaussianNB                              
            t2=time()
            clf = GaussianNB()                                                          # Initializing the Naive Bayes Classifier
            clf.partial_fit(x_train, y_train, np.unique(y_train))                       # Fitting the classifier to the training and testing the Naive Bayes Classifier
            y_pred = clf.predict(x_test)
            print(y_pred)
            t = time() - t2
            title = "Learning Curves (Naive Bayes)"
            geterror(x_train,y_train,clf,title);
        
            print("Training Complete")
        
        elif string=="K-Nearesr Neighbour":
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            print(y_pred)
            print("Training Complete")

        elif string=="Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            t3=time()
            clf=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=1)   # Initializing the Logistic Regression Classifier
            clf.fit(x_train,y_train)
            y_pred = clf.predict(x_test)
            print(y_pred)
            t = time() - t3
            title = "Learning Curves (Logistic Regression)"
            geterror(x_train,y_train,clf,title);
            print("Training Complete")

            t=str(t)
    
        classre=classification_report(y_test,y_pred)                       # Generating Classification Report
        T.insert(END,classre[1:5]+classre[1:32]+classre[1:13]+classre[60:90]+classre[1:11]+classre[1:2]+classre[115:140]+classre[1:7]+classre[161:195]) # Printing Precision and Recall Results 
        print(classre)
        confmat=confusion_matrix(y_test,y_pred)                                        # Calculating the Confusion Matrix for the classification
        T1.insert(END, confmat)
        T.insert(END, classre[1:9])
        T.insert(END, "Accuracy")
        T.insert(END, classre[1:5])
        T.insert(END, int(float((y_test==y_pred).sum())/len(y_test.T)*100))
        T.insert(END, "%")
        T.insert(END, classre[1:10]+classre[1:10])
        T.insert(END, "Class. Time")
        T.insert(END, classre[1:8])
        T.insert(END, t[0:4]+" sec")
        import matplotlib.pyplot as plt
    def process_dataset(string):

        
    
        if string=="Cleveland":
        
            heart = pnd.read_csv('processed.cleveland.data', names=header_row)          # Reading the dataset file in .data format using Pandas library function read_csv()
            print("Unprocessed Cleveland Dataset")
            print("************************************************************************")
            print(heart.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            import numpy as np
            has_hd_check = heart['diagnosis'] > 0                                                           # Getting the indices of individuals having heart disease
            has_hd_patients = heart[has_hd_check]
            heart['vessels'] = heart['vessels'].apply(lambda vessels: 0.0 if vessels == "?" else vessels)   # Replacing the unknown values in the dataset with float
            heart['vessels'] = heart['vessels'].astype(float)
            heart['thal'] = heart['thal'].apply(lambda thal: 0.0 if thal == "?" else thal)
            heart['thal'] = heart['thal'].astype(float)
            heart['diag_int'] = has_hd_check.astype(int)

            ind1 = np.where((heart['diagnosis'] == 1)|(heart['diagnosis'] ==2));
            ind2 = np.where((heart['diagnosis'] == 3)|(heart['diagnosis'] ==4));

            temp = heart['diagnosis'];
            temp.ix[ ind1 ] = 1;
            temp.ix[ ind2 ] = 2;
            heart['diagnosis'] = temp;

            global x_train
            global y_train
            global x_test
            global y_test
            x_train, x_test, y_train, y_test = train_test_split(heart.loc[:, 'age':'thal'], heart.loc[:, 'diagnosis'],   # Splitting the processed data into training data and testing data
                                                        test_size=0.30, random_state=42)                            # test_size = percent of data used for testing,
                                                                                                                    # random_state = for initializing the random number generator

            print("Processed Cleveland Dataset")
            print("************************************************************************")
            print(heart.loc[:, 'age':'diagnosis'])
            print("************************************************************************")
        

        elif string=="VA":

            import numpy as np


        
            heart_va = pnd.read_csv('processed.va.data', names=header_row)
            print("Unprocessed VA Dataset")
            print("************************************************************************")
            print(heart_va.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            has_hd_check = heart_va['diagnosis'] > 0

        
        
            heart_va['diag_int'] = has_hd_check.astype(int) 
            heart_va = heart_va.replace(to_replace='?', value=0.0)
            heart_va['diag_int'] = has_hd_check.astype(int)
        
            ind1 = np.where((heart_va['diagnosis'] == 1)|(heart_va['diagnosis'] ==2));
            ind2 = np.where((heart_va['diagnosis'] == 3)|(heart_va['diagnosis'] ==4));

            temp = heart_va['diagnosis'];
            temp.ix[ ind1 ] = 1;
            temp.ix[ ind2 ] = 2;

            heart_va['diagnosis'] = temp;
        
            print("Processed VA Dataset")
            print("************************************************************************")
            print(heart_va.loc[:, 'age':'diagnosis'])
            print("************************************************************************")
         
            x_train, x_test, y_train, y_test = train_test_split(heart_va.loc[:, 'age':'thal'], heart_va.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)
        
        elif string=="Hungarian":
            import numpy as np
            heart_hu = pnd.read_csv('processed.hungarian.data', names=header_row)
            print("Unprocessed Hungarian Dataset")
            print("************************************************************************")
            print(heart_hu.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            has_hd_check = heart_hu['diagnosis'] > 0
            heart_hu['diag_int'] = has_hd_check.astype(int)
            heart_hu = heart_hu.replace(to_replace='?', value=0.0)

            ind1 = np.where((heart_hu['diagnosis'] == 1)|(heart_hu['diagnosis'] ==2));
            ind2 = np.where((heart_hu['diagnosis'] == 3)|(heart_hu['diagnosis'] ==4));

            temp = heart_hu['diagnosis'];
            temp.ix[ ind1 ] = 1;
            temp.ix[ ind2 ] = 2;
            heart_hu['diagnosis'] = temp;

            print("Processed Hungarian Dataset")
            print("************************************************************************")
            print(heart_hu.loc[:, 'age':'diagnosis'])
            print("************************************************************************")
            heart_hu['diag_int'] = has_hd_check.astype(int)

        
            x_train, x_test, y_train, y_test = train_test_split(heart_hu.loc[:, 'age':'thal'], heart_hu.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)

        elif string=="all":
            import numpy as np
            heart_cl = pnd.read_csv('processed.cleveland.data', names=header_row)
            print("Unprocessed Cleveland Dataset")
            print("************************************************************************")
            print(heart_cl.loc[:, 'age':'diagnosis'])
            print("************************************************************************")
            has_hd_check = heart_cl['diagnosis'] > 0
            has_hd_patients = heart_cl[has_hd_check]
            heart_cl['diag_int'] = has_hd_check.astype(int)
            heart_cl['vessels'] = heart_cl['vessels'].apply(lambda vessels: 0.0 if vessels == "?" else vessels)
            heart_cl['vessels'] = heart_cl['vessels'].astype(float)
            heart_cl['thal'] = heart_cl['thal'].apply(lambda thal: 0.0 if thal == "?" else thal)
            heart_cl['thal'] = heart_cl['thal'].astype(float)

            ind1 = np.where((heart_cl['diagnosis'] == 1)|(heart_cl['diagnosis'] ==2));
            ind2 = np.where((heart_cl['diagnosis'] == 3)|(heart_cl['diagnosis'] ==4));

            temp = heart_cl['diagnosis'];
            temp.ix[ ind1 ] = 1;
            temp.ix[ ind2 ] = 2;
            heart_cl['diagnosis'] = temp;

            heart_va = pnd.read_csv('processed.va.data', names=header_row)
            print("Unprocessed VA Dataset")
            print("************************************************************************")
            print(heart_va.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            has_hd_check = heart_va['diagnosis'] > 0
            heart_va['diag_int'] = has_hd_check.astype(int)
            heart_va = heart_va.replace(to_replace='?', value=0.0)

            ind1 = np.where((heart_va['diagnosis'] == 1)|(heart_va['diagnosis'] ==2));
            ind2 = np.where((heart_va['diagnosis'] == 3)|(heart_va['diagnosis'] ==4));

            temp = heart_va['diagnosis'];
            temp.ix[ ind1 ] = 1;
            temp.ix[ ind2 ] = 2;
            heart_va['diagnosis'] = temp;

            print("Processed VA Dataset")
            print("************window1************************************************************")
            print(heart_va.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            heart_hu = pnd.read_csv('processed.hungarian.data', names=header_row)
            print("Unprocessed Hungarian Dataset")
            print("************************************************************************")
            print(heart_hu.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            has_hd_check = heart_hu['diagnosis'] > 0
            heart_hu['diag_int'] = has_hd_check.astype(int)
            heart_hu = heart_hu.replace(to_replace='?', value=0.0)

            ind1 = np.where((heart_hu['diagnosis'] == 1)|(heart_hu['diagnosis'] ==2));
            ind2 = np.where((heart_hu['diagnosis'] == 3)|(heart_hu['diagnosis'] ==4));

            temp = heart_hu['diagnosis'];
            temp.ix[ ind1 ] = 1;
            temp.ix[ ind2 ] = 2;
            heart_hu['diagnosis'] = temp;

            print("Processed Hungarian Dataset")
            print("************************************************************************")
            print(heart_hu.loc[:, 'age':'diagnosis'])
            print("************************************************************************")

            x_train1, x_test1, y_train1, y_test1 = train_test_split(heart_cl.loc[:, 'age':'thal'], heart_cl.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)
            x_train2, x_test2, y_train2, y_test2 = train_test_split(heart_va.loc[:, 'age':'thal'], heart_va.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)
            x_train3, x_test3, y_train3, y_test3 = train_test_split(heart_hu.loc[:, 'age':'thal'], heart_hu.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)

            # Combining the dataset for Cleveland, VA and Hungarian Dataset
            x_train4= x_train1.append(x_train2);
            x_train = x_train4.append(x_train3);
        
            y_train4 = y_train1.append(y_train2);
            y_train = y_train4.append(y_train3);

            x_test4 = x_test1.append(x_test2);
            x_test = x_test4.append(x_test3)

            y_test4 = y_test1.append(y_test2);
            y_test = y_test4.append(y_test3);
        

    button = Button(master, text="Process Dataset",height=1,fg="black",font=('algerian',13,'bold'),bg="violet",justify='center', command=lambda: process_dataset(var.get())) #Defining the button in the Tkinter Widget
    button.place(x=700,y=80)

    var1 = StringVar(master)
    var1.set("Select Classifier") # initial value

    option1 = OptionMenu(master, var1, "SVM", "Naive Bayes", "Logistic Regression","K-Nearesr Neighbour")
    option1.place(x=500,y=120)
    option1.config(bg = "violet")
    option1.config(fg = "black")
    option1.config(font=('algerian',10,'bold'))
    option1.config(width=12)
    #option.place ( relx=0.5, rely=0.1)
    button1 = Button(master, text=" Train Classifier",height=1,fg="black",font=('algerian',13,'bold'),bg="violet",justify='center', command=lambda: train_classifier(x_train,x_test,y_train,y_test,var1.get()))
    button1.place(x=700,y=120)




    #e1.bind('<Button-1>',e1.delete(0,END))

    def predres(clf):                                                                           # Defining function to predict the result from the user input data

        age= E1.get()
        sex = E2.get()
        pai= E3.get()
        bp= E4.get()
        chol = E5.get()
        fbs = E6.get()
        ecg = E7.get()
        maxhr = E8.get()
        eiang = E9.get()
        eist = E10.get()
        slope = E11.get()
        vessels = E12.get()
        thal = E13.get()
        pana=E0.get()
        

        aa = mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", db="cardiac1")
        mm = aa.cursor()
        
        #mm.execute("""INSERT INTO cardiac1 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (age,sex,pai,bp,chol,fbs,ecg,maxhr,eiang,eist,slope,vessels,thal,pana))
        #aa.commit()
        
        E14=E10.get()                                                                           # Converting the Eist data according to sign and decimal point
        if len(E14)==3 or len(E14)==4:
            E15=float(E14)
        else:
            E16=E14+'.0'
            E15=float(E16)
        
        test=[float(E1.get()+'.0'),float(E2.get()+'.0'),float(E3.get()+'.0'),float(E4.get()+'.0'),float(E5.get()+'.0'),float(E6.get()+'.0'),float(E7.get()+'.0'),float(E8.get()+'.0'),float(E9.get()+'.0'),E15,float(E11.get()+'.0'),float(E12.get()+'.0'),float(E13.get()+'.0')]
        test=np.reshape(test,(1,-1))
        #print(clf)
        
        
        
        
        if clf.predict(test) == 1:
        
            Labx1=Label(master,text="The Person has Mild cardiac arrhymia Disease", bg='orange')
            predict="The Person has Mild cardiac arrhymia Disease"
            mm.execute("""INSERT INTO cardiac1 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (age,sex,pai,bp,chol,fbs,ecg,maxhr,eiang,eist,slope,vessels,thal,pana,predict))
            aa.commit()
            Labx1.visible=False
            Labx1.place(x=600,y=430)
            Labx1.visible=True
            #T3.insert(END,"The Person has Heart Disease")

        elif clf.predict(test) == 2:
            Labx1=Label(master,text="The Person has Severe cardiac arrhymia Disease", bg='red')
            predict="The Person has Severe cardiac arrhymia Disease"
            mm.execute("""INSERT INTO cardiac1 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (age,sex,pai,bp,chol,fbs,ecg,maxhr,eiang,eist,slope,vessels,thal,pana,predict))
            aa.commit()
            Labx1.visible=False
            Labx1.place(x=600,y=430)
            Labx1.visible=True
        
        else:
            Labx1=Label(master,text="The Person does not have cardiac arrhymia Disease", bg='green')
            predict="The Person does not have cardiac arrhymia Disease"
            mm.execute("""INSERT INTO cardiac1 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (age,sex,pai,bp,chol,fbs,ecg,maxhr,eiang,eist,slope,vessels,thal,pana,predict))
            aa.commit()
            Labx1.visible=False
            Labx1.place(x=600,y=430)
            Labx1.visible=Truewindow1
       
        
        #T3.insert(END,"The Person does not have Heart Disease")

    def geterror(x_train,y_train,clf,title):
        #global outclass
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
        from sklearn.model_selection import ShuffleSplit
        clas=[];
    
        def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
            plt.figure()
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1,
                color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

            plt.legend(loc="best")
            plt.axis([ 0,len(y),0,1.1])
            return plt

    
        for index in range(len(y_train)):
            #x_train1=np.reshape(x_train.iloc[index:],(1,-1));
            x_train1=x_train.iloc[index].values.reshape(1,-1);
            outclass = clf.predict(x_train1);
            clas.append(outclass[0]);
        
        ind3 = np.where((clas == y_train));
        ind4 = np.where((clas != y_train));
        l=0
        m=0
    
        cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
    
        plot_learning_curve(clf, title, x_train, y_train, (0.7, 1.01), cv=cv, n_jobs=1)
    
        plt.show()
    
    
    button2 = Button(master, text=" Predict Heart Disease ",width=20,height=1,fg="black",font=('algerian',13,'bold'),bg="violet",justify='center',command=lambda:predres(clf))
    button2.place(x=600,y=300)
    

    btn6=Button(master,text="LOGOUT",width=8,height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=cardes)
    btn6.place(x=1100,y=80)
    #button2.configure(width=14)
    #button1.place(relx=0.1,rely=0.2)
    

    master.mainloop()


def adminlogin():
    def adminlogininto():
        usernames = e1.get()
        passwords = e2.get()
        if e1.get() == "" or e2.get() == "":
            tkMessageBox.showinfo("sorry","Please complete the required field")
        elif e1.get() == "admin" or e2.get() == "admin":
            tkMessageBox.showinfo("yeh","logged in")
            admindes()
        else:
            tkMessageBox.showinfo("Sorry" , "Wrong Password")
    global window1
    window1=Tk()
    window1.title("LOGIN PAGE")
    
    window1.geometry('700x500')
    image = Image.open('photo.jpg')
    image = image.resize((700, 600))
    photo_image = ImageTk.PhotoImage(image)
    label = Label(window1, image = photo_image)
    label.place(x=0,y=0)
    
    '''lb1=Label(window1,text="USERNAME",font=('algerian',25,'bold'),fg="BLACK",anchor='w')
    lb1.place(x=150,y=400)'''

    e1=Entry(window1,width=10,font=("bold",17),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e1.place(x=250,y=150)

    '''lb2=Label(window1,text="PASSWORD",font=('algerian',25,'bold'),fg="BLACK",anchor='w')
    lb2.place(x=150,y=450)'''

    e2=Entry(window1,width=10,show="*",font=("bold",17),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e2.place(x=250,y=200)

    btn6=Button(window1,text="LOGIN",width=8,height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=adminlogininto)
    btn6.place(x=270,y=300)
    btn6=Button(window1,text="REPORT",width=8,height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=data)
    btn6.place(x=350,y=350)
    
    window1.mainloop()

def data():
    def data1():
        aa = mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", db="cardiac1")
        mm = aa.cursor()
        pana = e11.get()
        print(pana)
        
        if e11.get() == "" :
            tkinter.messagebox.showinfo("sorry", "Please complete the required field")
        else:
            sql = "SELECT * FROM cardiac1 WHERE pana = '%s'"%(e11.get())
            mm.execute(sql)
            for i in pana:
                print( 0 )

            window = Tk()
            window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
            scrollbar = Scrollbar(window)
            scrollbar.pack(side=RIGHT, fill=Y)
            result=mm.fetchone()
            listbox = Listbox(window)
            listbox.pack()
            name=["age","sex","pai","bp","chol","fbs","ecg","maxhr","eiang","eist","slope","vessels","thal","patientName","Result"]
            for i in result:
                listbox.insert(END,i)
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)
            mainloop()
            window.withdraw()
            window.quit()
    window2 = Toplevel()
    window2.geometry('900x600')
    image=Image.open('download.jpg')
    image=image.resize((900,600))
    photo_image=ImageTk.PhotoImage(image)
    label=Label(window2,image=photo_image)
    label.place(x=0,y=0)

    e11= Entry(window2,width=15,font=("bold",17),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e11.place(x=420, y=420)

    btn = Button(window2, text="generate", width=8, height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=data1)
    btn.place(x=440, y=500)
    
    window2.mainloop()

def work():
    global master
    master1 = Tk()         # Defining the Tkinter widget
    master1.geometry('1500x600')
    lb1=Label(master1,text="USERNAME",font=('algerian',25,'bold'),fg="BLACK",anchor='w')
    lb1.place(x=150,y=400)

    e1=Entry(master1,width=10,font=("bold",17),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e1.place(x=250,y=150)

    btn6=Button(master1,text="REPORT",width=8,height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=data)
    btn6.place(x=270,y=300)

def admindes():
    window1.destroy()
    cardiac()

def cardes():
    master.destroy()
    adminlogin()
    
if __name__ == "__main__":
    adminlogin()
