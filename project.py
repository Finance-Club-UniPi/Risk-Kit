import yfinance as yf
import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as st
Di=yf.Ticker("KO").history(start="1990-01-01",interval="1wk",auto_adjust=True)["Close"]
Dm=yf.Ticker("^SP500TR").history(start="1990-01-01",interval="1wk")["Close"] #αποθηκευση δεδομενων της αγορας και της μετοχης στην τιμη κλεισιματος
def oikonmetr(Di,Dm):
 ri=[]
 rm=[]
 leng=len(Dm) #μεγεθος 
 i=0
 while i<leng-1:
  ri.append((Di[i+1]-Di[i])/Di[i])
  rm.append((Dm[i+1]-Dm[i])/Dm[i]) #υπολογισμος των αποδοσεων της αγορας και της μετοχης
  i+=1
 ci=[]
 cm=[]
 for i in range(leng-1):
  ci.append(ri[i]-2/5200)
  cm.append(rm[i]-2/5200) #υπολογισμος της διαφορας μεταξυ αποδοσεων και του μηδενικου επιτοκιου
 sumx=0
 sumy=0
 for i in range(leng-1):
  sumx+=cm[i]
  sumy+=ci[i]
 moy=sumy/(leng-1)
 mox=sumx/(leng-1) #υπολογισμος μεσης τιμης
 sum1=0
 sum4=0
 for i in range(leng-1):
  sum1+=(cm[i]-mox)**2
  sum4+=(ci[i]-moy)**2
 sx=sum1/(leng-2)
 sy=sum4/(leng-2) #υπολογισμος δειγματικης διακυμανσης
 sum2=0
 for i in range(leng-1):
  sum2+=(cm[i]-mox)*(ci[i]-moy) #υπολογισμος της δειγματικης συνδιακυμανσης 
 cov=sum2
 b=cov/sum1
 a=moy-b*mox #υπολογισμος του α και β στην γραμμικη παλινδρομηση και του μοντελου capm
 e=[]
 for i in range(leng-1):
  e.append(ci[i]-a-b*cm[i]) #υπολογισμος του λαθους
 sum3=0
 for i in range(leng-1):
  sum3+=e[i]**2
 sb=np.sqrt(sum3/(leng-3)/sum1) #υπολογισμος της δειγματικης τυπικης αποκλισης του β
 t=st.t.ppf(1-0.025,leng-3)
 pl.scatter(cm,ci,color="red") #εμφανιση των σημειων
 cm1=np.min(cm)
 cm2=np.max(cm)
 pl.plot([cm1,cm2],[a+b*cm1,a+b*cm2],color="blue")
 pl.xlabel("market returns")
 pl.ylabel("coca-cola returns")
 pl.title("coca-cola linear regression")
 pl.show() #εμφανιση και της γραμμης
 print(b-t*sb,b+t*sb) #υπολογισμος του διαστηματος εμπιστοσυνης του β
 if 0<b-t*sb or 0>b+t*sb:
  print("Denied")
 else:
  print("good enough") #υπολογισμος του ελεγχου υποθεσης οτι το πραγματικο β ειναι 0 με 95% βεβαιοτητα
 t=st.t.ppf(1-0.05,leng-3)
 if t*sb+b>1:
  print("denied")
 else:
  print("good enough") #υπολογισμος του ελεγχου υποθεσης οτι το πραγματικο β ειναι μικροτερο απο 1
 R2=1-(sum3/(leng-3)/sy)
 print(R2) #υπολογισμος του ποσοστου που η διακυμανση της αγορας εξηγει την διακυμανση της μετοχης
oikonmetr(Di,Dm)

 

