import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Hourly
import numpy as np
from datetime import timedelta
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def prepare_data(df, weather=False):

    df["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in df["timestamp"].values]


    df["day_of_week"] = [x.day_of_week for x in df.timestamp]
    df["pon"] = df["day_of_week"] == 0
    df["pet"] = df["day_of_week"] == 4
    df["vikend"] = df["day_of_week"].isin([5,6])
    df = df.drop(['day_of_week'], axis=1)


    df["time_int"] = [int(ts.hour * 60 + ts.minute) for ts in df["timestamp"]]


    pocitnice = ["26. 6. 2022", "27. 6. 2022", "28. 6. 2022", "29. 6. 2022", "30. 6. 2022", "1. 7. 2022", "2. 7. 2022", "3. 7. 2022", "4. 7. 2022", "5. 7. 2022", "6. 7. 2022", "7. 7. 2022", "8. 7. 2022", "9. 7. 2022", "10. 7. 2022", "11. 7. 2022", "12. 7. 2022", "13. 7. 2022", "14. 7. 2022", "15. 7. 2022", "16. 7. 2022", "17. 7. 2022", "18. 7. 2022", "19. 7. 2022", "20. 7. 2022", "21. 7. 2022", "22. 7. 2022", "23. 7. 2022", "24. 7. 2022", "25. 7. 2022", "26. 7. 2022", "27. 7. 2022", "28. 7. 2022", "29. 7. 2022", "30. 7. 2022", "31. 7. 2022", "1. 8. 2022", "2. 8. 2022", "3. 8. 2022", "4. 8. 2022", "5. 8. 2022", "6. 8. 2022", "7. 8. 2022", "8. 8. 2022", "9. 8. 2022", "10. 8. 2022", "11. 8. 2022", "12. 8. 2022", "13. 8. 2022", "14. 8. 2022", "15. 8. 2022", "16. 8. 2022", "17. 8. 2022", "18. 8. 2022", "19. 8. 2022", "20. 8. 2022", "21. 8. 2022", "22. 8. 2022", "23. 8. 2022", "24. 8. 2022", "25. 8. 2022", "26. 8. 2022", "27. 8. 2022", "28. 8. 2022", "29. 8. 2022", "30. 8. 2022", "31. 8. 2022"] + ["31. 10. 2022", "1. 11. 2022", "2. 11. 2022", "3. 11. 2022", "4. 11. 2022", "25. 12. 2022", "26. 12. 2022", "27. 12. 2022", "28. 12. 2022", "29. 12. 2022", "30. 12. 2022", "31. 12. 2022", "1. 1. 2023", "2. 1. 2023", "6. 2. 2023", "7. 2. 2023", "8. 2. 2023", "9. 2. 2023", "10. 2. 2023", "10. 4. 2023", "26. 4. 2023", "27. 4. 2023", "28. 4. 2023", "29. 4. 2023", "30. 4. 2023", "1. 5. 2023", "2. 5. 2023", "25. 6. 2023"]
    pocitnice = [datetime.strptime(x,"%d. %m. %Y").date() for x in pocitnice]
    df["date"] = [ts.date() for ts in df["timestamp"]]
    df["pocitnice"] = df["date"].isin(pocitnice)
    df = df.drop(['date'], axis=1)
    
    if weather:
        start = datetime(2022, 1, 1)
        end = datetime(2022, 12, 31)

        location = Point(46.056946, 14.505751, 70)

        data = Hourly(location, start, end)
        data = data.fetch()

        data = data.reset_index()
        data["timestamp_round"] = data["time"]

        data = data[["temp","prcp","timestamp_round"]]

        def hour_rounder(t):
            return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                    +timedelta(hours=t.minute//30)) 

        # glej vreme pred 1 uro
        df["timestamp_round"] = [hour_rounder(ts) - timedelta(hours=1) for ts in df["timestamp"]]

        df = df.merge(data, on='timestamp_round', how='left')
        df = df.drop(['timestamp_round'], axis=1)
    return df

def opremi_z_urami(df, legal_values, ure=[1,1.5,2], tol = 10):
    for idt, t in enumerate(ure):
        l=[]
        for index, row in df.iterrows():
                target = row['timestamp'] - timedelta(hours=t)
                out = None
                targetminus = target
                targetplus = target
                for i in range(tol):
                    targetminus -= timedelta(minutes=1)
                    targetplus += timedelta(minutes=1)
                    if targetminus in legal_values:
                        out = targetminus
                        break
                    if targetplus in legal_values:
                        out = targetplus
                        break
                l.append(out)
        df["timestampminus"+str(t)] = l
    return df


def init(ure=[1,1.5,2],weather=False):
    

    df = pd.read_csv("bicikelj_train.csv")
    df = prepare_data(df,weather=weather)
    dt = pd.read_csv("bicikelj_test.csv")
    dt = prepare_data(dt,weather=weather)

    df = opremi_z_urami(df,legal_values = df["timestamp"].values,ure=ure,tol=10)
    dt = opremi_z_urami(dt,legal_values = df["timestamp"].values,ure=ure,tol=200) # sam da zafilamo
    
    return df, dt

def get_pp(df,sourse_of_history, postaja="PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"):
    attrs = [postaja] + [x for x in list(df.columns) if x.islower()]
    for x in ["date", "time", "timestamp_round","day_of_week"]:
        if x in attrs: attrs.remove(x)

    dg = df[attrs]
    dg.columns = ["y"] + list(dg.columns)[1:]
    
    dj = sourse_of_history[attrs]
    dj.columns = dg.columns    
    
    dh = dj[["y","timestamp"]]
    
    tss = [x[len("timestampminus"):] for x in df.columns if "timestampminus" in x]
    
    to_use_mask = ~np.isnan(dg["timestampminus"+tss[0]])
    for t in tss[1:]:
        to_use_mask = to_use_mask & ~np.isnan(dg["timestampminus"+t])
    to_use = dg[to_use_mask]
    pp = to_use
    
    ##################33
    for t in tss:
        a = "y"+str(t)
        b = "timestampminus"+str(t)
        
        dh.columns = [a,b]
        
        pp = pd.merge(pp, dh, on=b,
                        how = 'left')
    

    pp = pp.drop(["timestamp"] + ["timestampminus"+str(t) for t in tss],axis=1)
    # pp = pp.drop(["timestamp","timestampminus1","timestampminus2"],axis=1)
    X = pp.iloc[:,1:]
    #X  = preprocessing.normalize(X)
    y = np.array(pp.iloc[:,0])
    return (pp,np.array(X),y)

# def get_pp(df,sourse_of_history, postaja="PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"):
#     attrs = [postaja] + [x for x in list(df.columns) if x.islower()]
#     for x in ["date", "time", "timestamp_round","day_of_week"]:
#         if x in attrs: attrs.remove(x)

#     dg = df[attrs]
#     dg.columns = ["y"] + list(dg.columns)[1:]
    
#     dj = sourse_of_history[attrs]
#     dj.columns = dg.columns    
    
#     dh = dj[["y","timestamp"]]
#     dh.columns = ["pred_eno_uro","timestampminus1"]
    
#     to_use = dg[~np.isnan(dg["timestampminus1"]) & ~np.isnan(dg["timestampminus2"])]
    
#     pp = pd.merge(to_use, dh, on="timestampminus1",
#                     how = 'left')
    
#     dh.columns = ["pred_dvema_urama","timestampminus2"]
#     pp = pd.merge(pp, dh, on="timestampminus2",
#                     how = 'left')
#     pp = pp.drop(["timestamp","timestampminus1","timestampminus2"],axis=1)
#     X = pp.iloc[:,1:]
#     #X  = preprocessing.normalize(X)
#     y = np.array(pp.iloc[:,0])
#     return (pp,np.array(X),y)

def run(df,dt,clf,inside=True,round=True):
    mse_ = 0
    postaje = [x for x in list(df.columns) if x.isupper()]
    for postaja in postaje:
        print(postaja)
        if inside:
            _,X,y = get_pp(df,df,postaja)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        else:
            _,X_train,y_train = get_pp(df,df,postaja)
            _,X_test,y_test = get_pp(dt,df,postaja)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        # y_pred = gbr(X_train,y_train,X_test)
        if inside:
            mse = mean_squared_error(y_test,y_pred)
            mse_ += mse
            print(mse)
        else:
            if round:
                dt[postaja] = np.round(y_pred)
            else:
                dt[postaja] = y_pred
        
    return dt, mse_

# def gbr(X_train,y_train,X_test):
#     # clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
#     clf = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     return y_pred
#     # #cross_val_score(clf, X_train, y_train, cv=5)
#     # print(y_pred)
#     # mse = mean_squared_error(y_test,y_pred)
#     # print(mse)

# def run(df,dt,inside=True,round=True):
#     postaje = [x for x in list(df.columns) if x.isupper()]
#     for postaja in postaje:
#         print(postaja)
#         if inside:
#             _,X,y = get_pp(df,df,postaja)
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
#         else:
#             _,X_train,y_train = get_pp(df,df,postaja)
#             _,X_test,y_test = get_pp(dt,df,postaja)
#         y_pred = gbr(X_train,y_train,X_test)
#         if inside:
#             mse = mean_squared_error(y_test,y_pred)
#             print(mse)
#         else:
#             if round:
#                 dt[postaja] = np.round(y_pred)
#             else:
#                 dt[postaja] = y_pred
        
#     return dt
