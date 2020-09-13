from tkinter import *
from tkinter import filedialog
from pandastable import Table
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox
from sklearn.impute import SimpleImputer
import webbrowser

root = Tk()

#callback function
def callback(url):
    webbrowser.open_new(url)
#function for browse window
def browse_window():
    bw = Toplevel()
    bw.geometry("700x700")
    link1 = Label(bw, text="Make sure to split your dataset into two parts by clicking on this link before moving forward",font=50, fg="blue", cursor="hand2")
    link1.pack()
    link1.bind("<Button-1>", lambda e: callback("https://www.splitcsv.com/"))
    label = Label(bw, text = "Choose the files which contains data",font = 100)
    label.pack()
    browse_button_frame = Frame(bw, borderwidth = 2, bg = "grey", relief = SUNKEN)
    browse_button_frame.pack(side = TOP)
    def browse():
        dataFile = filedialog.askopenfilenames(initialdir="/", title = "Select the file which contains data(max 2 files)", filetypes = (("csv file","*.csv"), ("All files","*.*")))
        for i in dataFile:
            label = Label(bw, text = i).pack()
        def continue_button():
            merge_window = Toplevel()
            merge_window.geometry("544x744")
            f1 = Frame(merge_window,width = 500, bg = "linen", borderwidth=6,relief=SUNKEN)
            f1.pack(side=LEFT,fill="y")
            f2 = Frame(merge_window,width = 500, bg = "linen", borderwidth=6,relief=SUNKEN)
            f2.pack(side=LEFT,fill="y")
            pt = Table(f1)
            pt.show()
            pt.importCSV(filename=dataFile[0], dialog=False)
            pt = Table(f2)
            pt.show()
            pt.importCSV(filename=dataFile[1], dialog=False)
            
            def merge_data():
                mergeWindow = Toplevel()
                mergeWindow.title("Merged Data Window")
                reader = csv.reader(open(dataFile[0]))
                reader1 = csv.reader(open(dataFile[1]))
                f = open("combined.csv", "w")
                writer = csv.writer(f)
                next(reader1)
                for row in reader:
                    writer.writerow(row)
                for row in reader1:
                    writer.writerow(row)
                f.close()
                frame = Frame(mergeWindow)
                frame.pack()

                pt = Table(frame)
                pt.show()

                pt.importCSV(filename='combined.csv',dialog = True)
                label = Label(mergeWindow, text = "This is merged file",font=50)
                label.pack()
                dataPreprocessingLabel = Label(mergeWindow,text = "Do you want data preprocessing on data?",font=100)
                dataPreprocessingLabel.pack()
                def yes_clicked():
                    #Here we have to do data preprocessing
                    dataPreprocessingWindow = Toplevel()
                    dataPreprocessingWindow.title("Data Preprocessing Window")
                    dataPreprocessingWindow.geometry("700x700")
                    missing_values_indexes = StringVar()
                    icategorical_values_indexes = StringVar()
                    dcategorical_values_indexes = StringVar()
                    def ok_clicked():
                        try:
                            listOfMissingValuesIndexes = []
                            listOfICategoricalValuesIndexes = []
                            listOfDCategoricalValuesIndexes = []
                            index = missing_values_indexes.get()
                            iindex = icategorical_values_indexes.get()
                            dindex = dcategorical_values_indexes.get()
                            for i in index:
                                if(i==' '):
                                    pass
                                else:
                                    ind = int(i)
                                    listOfMissingValuesIndexes.append(ind)
                            for i in iindex:
                                if(i==' '):
                                    pass
                                else:
                                    ind = int(i)
                                    listOfICategoricalValuesIndexes.append(ind)
                            for i in dindex:
                                if(i==' '):
                                    pass
                                else:
                                    ind = int(i)
                                    listOfDCategoricalValuesIndexes.append(ind)
                                                        
                            print(listOfMissingValuesIndexes)
                            print(listOfICategoricalValuesIndexes)
                            print(listOfDCategoricalValuesIndexes)
                        except:
                            print("Your data doesn't have any categorical value")
                        finally:
                            import numpy as np
                            import matplotlib.pyplot as plt
                            import pandas as pd


                            dataset = pd.read_csv('combined.csv')
                            X = dataset.iloc[:, :-1].values
                            y = dataset.iloc[:, -1].values


                            print(X)
                            print(y)
                            try:
                                from sklearn.impute import SimpleImputer
                                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                                imputer.fit(X[:, listOfMissingValuesIndexes])
                                X[:, listOfMissingValuesIndexes] = imputer.transform(X[:, listOfMissingValuesIndexes])

                                print(X)

                                from sklearn.compose import ColumnTransformer
                                from sklearn.preprocessing import OneHotEncoder
                                ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), listOfICategoricalValuesIndexes)], remainder='passthrough')
                                X = np.array(ct.fit_transform(X))
                                print(X)

                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                y = le.fit_transform(y)
                                print(y)
                            except:
                                print("No need")
                            finally:

                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
                                
                                Label(dataPreprocessingWindow,text = "This is metrices of features X after data preprocessing",bg ="orange", font=50).grid()
                                Label(dataPreprocessingWindow,text = X,font=50).grid()
                                Label(dataPreprocessingWindow,text = "This is dependent variable y after data preprocessing",bg = "orange",font=50).grid()
                                Label(dataPreprocessingWindow,text = y,font=50).grid()
                                Label(dataPreprocessingWindow,text = "Data set splitted into training set(80%) and test set(20%)",bg ="orange", font=70).grid()

                        def move_forward_clicked():
                            chooseAlgoWindow = Toplevel()
                            chooseAlgoWindow.title("Choice is hard")
                            chooseAlgoWindow.geometry("700x700")
                            Label(chooseAlgoWindow,text = "Choose what you want to do on your data(REGRESSION, CLASSIFICATION OR CLUSTERING)",bg = "antique white",font=80).grid(row=0,column=0)
                            def regression():
                                def linear_regression():
                                    from sklearn.linear_model import LinearRegression
                                    regressor = LinearRegression()
                                    regressor.fit(X_train, y_train)
                                    #Predicting the test set results
                                    y_pred = regressor.predict(X_test)
                                    #Visualizing the training set results
                                    def train_set():
                                        plt.scatter(X_train[:,0], y_train, color = 'red')
                                        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
                                        plt.title('Training set')
                                        plt.show()
                                    #Visualizing the test set results
                                    def test_set():
                                        plt.scatter(X_test[:,0], y_test, color = 'red')
                                        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
                                        plt.title("Test set")
                                        plt.show()
                                    #Predicting new value
                                    def predict_new_value():
                                         new_value_predict = IntVar()
                                         Label(chooseAlgoWindow,text = "Enter the value to predict",font=70).grid()
                                         Entry(chooseAlgoWindow,textvariable = new_value_predict,width=50).grid()
                                         def predict_clicked():
                                             value_to_predict = new_value_predict.get()
                                             X_predict = [value_to_predict]  
                                             y_predict = regressor.predict([X_predict])
                                             Label(chooseAlgoWindow,text = "Predicted value is :-",font=100).grid()
                                             Label(chooseAlgoWindow,text = y_predict,font = 150).grid()
                                         Button(chooseAlgoWindow,text = "Predict",fg = "orange",command = predict_clicked,cursor = "hand2").grid()
                                         
                                        
                                    Button(chooseAlgoWindow,text = "Click to visualize the train test result",bg = "orange",fg = "black",command = train_set,cursor = "hand2").grid()
                                    Button(chooseAlgoWindow,text = "Click to visualize the test test result",bg = "orange",fg = "black",command = test_set,cursor = "hand2").grid()
                                    Button(chooseAlgoWindow,text = "Click here to predict new value",bg = "orange",fg = "black",command = predict_new_value,cursor = "hand2").grid()
                                def multiple_linear_regression():
                                    from sklearn.linear_model import LinearRegression
                                    regressor = LinearRegression()
                                    regressor.fit(X_train, y_train)
                                    #Predicting the Test set results
                                    y_pred = regressor.predict(X_test)
                                    np.set_printoptions(precision=2)
                                    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
                                    Label(chooseAlgoWindow,text = "Here is the predicted test set result(FIRST COLUMN = TEST SET VALUE, SECOND COLUMN = PREDICTED VALUE)",bg = "orange").grid()
                                    Label(chooseAlgoWindow,text = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)).grid()

                        
                                    
                                Label(chooseAlgoWindow,text = "Ok...Which algo??",bg = "orange",font=40).grid(row=4,column=1)
                                Button(chooseAlgoWindow,text = "1.Simple Linear Regression",bg = "linen",fg="black",command = linear_regression,cursor="hand2").grid(row=5,column=1)
                                Button(chooseAlgoWindow,text = "2.Multiple Linear Regression",bg = "linen",fg="black",command = multiple_linear_regression,cursor="hand2").grid(row=6,column=1)
                                Button(chooseAlgoWindow,text = "3.Polynomial Regression",bg = "linen",fg="black",cursor="hand2").grid(row=7,column=1)
                                Button(chooseAlgoWindow,text = "4.Support Vector Regression",bg = "linen",fg="black",cursor="hand2").grid(row=8,column=1)
                                Button(chooseAlgoWindow,text = "5.Decision Tree Regression",bg = "linen",fg="black",cursor="hand2").grid(row=9,column=1)
                                Button(chooseAlgoWindow,text = "6.Random Forest Regression",bg = "linen",fg="black",cursor="hand2").grid(row=10,column=1)
                            def classification():
                                Label(chooseAlgoWindow,text = "Ok...Which algo??",bg = "orange",font=40).grid(row=4,column=2)
                                Button(chooseAlgoWindow,text = "1.Logistic Regression",bg = "linen",fg="black",cursor="hand2").grid(row=5,column=2)
                                Button(chooseAlgoWindow,text = "2.K-Nearest Neighbours",bg = "linen",fg="black",cursor="hand2").grid(row=6,column=2)
                                Button(chooseAlgoWindow,text = "3.Support Vector Machine",bg = "linen",fg="black",cursor="hand2").grid(row=7,column=2)
                                Button(chooseAlgoWindow,text = "4.Kernel SVM",bg = "linen",fg="black",cursor="hand2").grid(row=8,column=2)
                                Button(chooseAlgoWindow,text = "5.Naive Byes",bg = "linen",fg="black",cursor="hand2").grid(row=9,column=2)
                                Button(chooseAlgoWindow,text = "6.Decision Tree Classification",bg = "linen",fg="black",cursor="hand2").grid(row=10,column=2)
                                Button(chooseAlgoWindow,text = "7.Random Forest Classification",bg = "linen",fg="black",cursor="hand2").grid(row=11,column=2)
                                
                            def clustering():
                                Label(chooseAlgoWindow,text = "Ok...Which algo??",bg = "orange",font=40).grid(row=4,column=3)
                                Button(chooseAlgoWindow,text = "1.K-Means Clustering",bg = "linen",fg="black",cursor="hand2").grid(row=5,column=3)
                                Button(chooseAlgoWindow,text = "2.Heirarchical Clustering",bg = "linen",fg="black",cursor="hand2").grid(row=6,column=3)
                            Button(chooseAlgoWindow,text = "REGRESSION",bg = "linen",fg = "black",height=2,width = 30,command = regression,cursor="hand2").grid(row=3,column=1)
                            Button(chooseAlgoWindow,text = "CLASSIFICATION",bg = "linen",fg = "black",height=2,width = 30,command = classification,cursor="hand2").grid(row=3,column=2)
                            Button(chooseAlgoWindow,text = "CLUSTERING",bg = "linen",fg = "black",height=2,width = 30,command = clustering,cursor="hand2").grid(row=3,column=3)
                        Button(dataPreprocessingWindow,text = "Move Forward",bg = "orange",fg = "linen",command = move_forward_clicked,cursor="hand2").grid()
                        
                    Label(dataPreprocessingWindow,text = "Enter indexes(space separated) that can contain missing data",font=70).grid(row=0,column=0)
                    Entry(dataPreprocessingWindow,textvariable = missing_values_indexes,width=50).grid()
                    Label(dataPreprocessingWindow,text = "Enter indexes(space separated) of independent variables that can contain categorical data",font=70).grid()
                    Entry(dataPreprocessingWindow,textvariable = icategorical_values_indexes,width=50).grid()
                    Button(dataPreprocessingWindow,text = "OK",command = ok_clicked,cursor="hand2").grid()
                 
                                        
                    
                        
                    
                def no_clicked():
                    messagebox.showinfo("Recommendation","We recommend you to do data preprocessing before moving forward!")
                    
                yesButton = Button(mergeWindow,text = "Yes(RECOMMENDED)",height = 2,width = 20,bg="linen",command = yes_clicked,cursor="hand2")
                yesButton.pack()
                noButton = Button(mergeWindow,text = "No",height = 2,width = 20,command = no_clicked,bg="linen",cursor="hand2")
                noButton.pack()
                Label(mergeWindow,text = "Make sure before clicking on YES that the dependent variable is the last column in data",font=100).pack()
                Label(mergeWindow,text = "Or if you want clustering on data then click the below button",font=100,bg="orange",fg="black").pack()
                def clusteringClicked():
                    Label(mergeWindow,text = "Which algo?? Choose the right algo according to the data",bg="green yellow").pack()
                    Button(mergeWindow,text = "K-Means Algorithm",bg="linen",fg="black",cursor="hand2").pack()
                    heirarchicalButton = Button(mergeWindow,text = "Heirarchical Clustering",bg="linen",fg = "black", cursor="hand2").pack()
                clusteringButton = Button(mergeWindow,text = "Click Here For Clustering",height = 2,width = 20,command=clusteringClicked,cursor="hand2")
                clusteringButton.pack()
            merge_button = Button(merge_window, text = "Merge the data of both files",bg  = "orange",fg = "green",height=2,width=30,command = merge_data,cursor="hand2")
            merge_button.pack(side = LEFT, anchor='n')
        
        continue_btn = Button(bw, text = "CONTINUE",height = 2, width = 30 , command = continue_button,bg="orange",cursor="hand2");
        continue_btn.pack(anchor = 'center')
        
    browse_button = Button(browse_button_frame, text = "Browse a file",height = 2, width = 30 ,bg="orange", command = browse,cursor="hand2");
    browse_button.pack()
    

root.geometry("1000x700")
root.title("RexD ML Tool")

photo = PhotoImage(file="KMLToolImage.png")
label = Label(image=photo)
label.pack(pady=150)


buttonFrame = Frame(root,borderwidth = 2, bg = "grey", relief = SUNKEN)
buttonFrame.pack(side = TOP)

startButton = Button(buttonFrame, text="START", fg="Green", bg = "orange", height = 2, width = 30,command = browse_window,cursor="hand2");
startButton.pack()
Label(root,text = "Click on start to upload data",bg="linen",font=70).pack()

root.mainloop()
