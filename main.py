import numpy as np
from flask import Flask,request,jsonify,render_template,url_for,send_file
import pickle
import pandas as pd
from datetime import datetime

#s="\\"+str(datetime.now())

app=Flask(__name__)
model=pickle.load(open(r"iris.pkl",'rb'))

# df = pd.concat(pd.read_excel(r'C:\Users\Ramu\Desktop\Machine Learning\datasets\iris_check.xlsx', sheet_name=None), ignore_index=True)

# y_pred=model.predict(df)


# @app.route('/')
# def home():
#     return render_template('index1.html')

    
@app.route('/',methods=["GET",'POST'])
def upload():
    
    if  request.method=='POST':
        #print("POST")
        
        data = pd.concat(pd.read_excel(request.files.get('file'), sheet_name=None), ignore_index=True)
        #data = pd.read_csv(request.files.get('file'))
        #print("data")
        y_pred=model.predict(data)
        
        y_pred=list(y_pred)
        
        flower_names=[]
        
        for z in y_pred:
            if z==0:
                flower_names.append("Iris Setosa")
            elif z==1:
                flower_names.append("Iris Versicolour")
            else:
                flower_names.append("Iris Virginica")
                
            
        y_df=pd.DataFrame({"Flower_names":flower_names})
        
        #print(y_df)
        
        data=pd.concat([data,y_df],axis=1)
        #print(data.shape)
        #data.to_excel(r'C:\Users\Ramu\Desktop\Machine Learning\gcp_iris'+s+'.xlsx',index=False)
        #print(data)
        #data.to_excel(r'data_iris.xlsx',index=False)
        return render_template('index1.html',shape=data.to_html())
    
    return render_template('index1.html')
    
# @app.route('/return_file',methods=["GET"])

# def return_file():
#     #return "hi"
#     try:
#         print("message")
#         return send_file('C:/Users/Ramu/Desktop/Machine Learning/gcp_iris/data_iris.xlsx',as_attachment=True)
        
#     except Exception as e:
#         print("line 58")
#         return str(e)
    
# @app.route("/file-downloads/")
# def file_downlaods():
#     try:
#         #print("line 64")
#         return render_template("complete.html")
#     except Exception as e:
#         #print("line 67")
#         return str(e)
       
if __name__=='__main__':
     app.run(debug=True)    