import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt


model_data = pd.read_csv('C:/Users/chandiramania/Desktop/Data/train_102919.csv', low_memory=False)
test_data = pd.read_csv('C:/Users/chandiramania/Desktop/Data/test_102919.csv', low_memory=False)
lookup_data = pd.read_csv('C:/Users/chandiramania/Desktop/Data/RNL Jan2019 Source Lookup.csv')
highschool_data = pd.read_csv('C:/Users/chandiramania/Desktop/Data/High_School_Address_Distance_Master_Database.csv')

highschool_data = highschool_data[['school', 'country', 'SUdistance']]
highschool_data = highschool_data.drop_duplicates(subset=['school', 'country'],keep='first')
lookup_data = lookup_data[['ISOURCE_CD', 'ISOURCE_GRP']]
model_data = pd.merge(model_data, lookup_data, how='left', on=['ISOURCE_CD'])
test_data = pd.merge(test_data, lookup_data, how='left', on=['ISOURCE_CD'])

selected_model_data = model_data[['STUDENTID','STATE', 'COUNTRY',
       'GENDER', 'FLAG_APP', 'DATE_INQ',
       'DATE_APP', 'FLAG_INTERNATIONAL', 'Citizenship',
       'FIRSTSOURCEAPP', 'MAJOR1', 'DEPT1', 'HSCODE',
       'Ugrd_Recruit_Ipeds_Ethnicity', 'CA_VISIT', 'CA_VISIT_DATE', 'ISOURCE_GRP','Ext Org Name']]
test_data = test_data[['STUDENTID','STATE', 'COUNTRY',
       'GENDER', 'DATE_INQ', 'DATE_APP',
       'FLAG_INTERNATIONAL', 'Citizenship',
       'FIRSTSOURCEAPP', 'MAJOR1', 'DEPT1', 'HSCODE',
       'Ugrd_Recruit_Ipeds_Ethnicity', 'CA_VISIT', 'CA_VISIT_DATE', 'ISOURCE_GRP', 'Ext Org Name', 'FLAG_APP']]

selected_model_data['static_date'] = pd.to_datetime("'2019-09-01'".replace("'",""))

selected_model_data['DATE_INQ'] = pd.to_datetime(selected_model_data['DATE_INQ'])
selected_model_data['CA_VISIT_DATE'] = pd.to_datetime(selected_model_data['CA_VISIT_DATE'])
selected_model_data['DATE_APP'] = pd.to_datetime(selected_model_data['DATE_APP'])

selected_model_data['Days_Difference'] = selected_model_data['static_date'] - selected_model_data['DATE_INQ']
#divide by np.timedelta64 to get number of days in float
selected_model_data['Days_Difference'] = selected_model_data['Days_Difference'] / np.timedelta64(1, 'D')


test_data['static_date'] = pd.to_datetime("'2020-09-01'".replace("'",""))

test_data['DATE_INQ'] = pd.to_datetime(test_data['DATE_INQ'])
test_data['CA_VISIT_DATE'] = pd.to_datetime(test_data['CA_VISIT_DATE'])
test_data['Days_Difference'] = test_data['static_date'] - test_data['DATE_INQ']
#divide by np.timedelta64 to get number of days in float
test_data['Days_Difference'] = test_data['Days_Difference'] / np.timedelta64(1, 'D')

selected_model_data = selected_model_data[selected_model_data['FIRSTSOURCEAPP'] != 'Y']
selected_model_data['CA_VISIT'] = np.where(selected_model_data['CA_VISIT_DATE'] > selected_model_data['DATE_APP'], 0,
                                           selected_model_data['CA_VISIT'])

selected_model_data = selected_model_data.fillna(999)
test_data = test_data.fillna(999)

selected_model_data['COUNTRY'] = np.where(selected_model_data['STATE'].isin(selected_model_data[selected_model_data['COUNTRY']=='United States']['STATE'].unique()),
                                 'United States', selected_model_data['COUNTRY'])
test_data['COUNTRY'] = np.where(test_data['STATE'].isin(test_data[test_data['COUNTRY']=='United States']['STATE'].unique()),
                                 'United States', test_data['COUNTRY'])

selected_model_data['COUNTRY'] = np.where(selected_model_data['COUNTRY']=='Korea, Republic of', 'South Korea', selected_model_data['COUNTRY'])
selected_model_data['COUNTRY'] = np.where(selected_model_data['COUNTRY']=='Viet Nam', 'Vietnam', selected_model_data['COUNTRY'])
selected_model_data['COUNTRY'] = np.where(selected_model_data['COUNTRY']=='Russian Federation', 'Russia', selected_model_data['COUNTRY'])
selected_model_data['COUNTRY'] = np.where(selected_model_data['COUNTRY']=='Myanmar', 'Myanmar (Burma)', selected_model_data['COUNTRY'])

test_data['COUNTRY'] = np.where(test_data['COUNTRY']=='Korea, Republic of', 'South Korea', test_data['COUNTRY'])
test_data['COUNTRY'] = np.where(test_data['COUNTRY']=='Viet Nam', 'Vietnam', test_data['COUNTRY'])
test_data['COUNTRY'] = np.where(test_data['COUNTRY']=='Russian Federation', 'Russia', test_data['COUNTRY'])
test_data['COUNTRY'] = np.where(test_data['COUNTRY']=='Myanmar', 'Myanmar (Burma)', test_data['COUNTRY'])

selected_model_data = pd.merge(selected_model_data, highschool_data, how='left',left_on = ['Ext Org Name', 'COUNTRY'], right_on=['school','country'])
test_data = pd.merge(test_data, highschool_data, how='left',left_on = ['Ext Org Name', 'COUNTRY'], right_on=['school','country'])
country_match = highschool_data[['country','SUdistance']].groupby(['country']).mean()
selected_model_data = selected_model_data.drop(['school', 'country'], axis=1)
test_data = test_data.drop(['school', 'country'], axis=1)
selected_model_data['country_distance']=selected_model_data.COUNTRY.map(country_match.SUdistance)
test_data['country_distance']=test_data.COUNTRY.map(country_match.SUdistance)
selected_model_data['SUdistance']=np.where(selected_model_data.SUdistance.isna(), selected_model_data['country_distance'],selected_model_data['SUdistance'])
test_data['SUdistance']=np.where(test_data.SUdistance.isna(), test_data['country_distance'],test_data['SUdistance'])

selected_model_data['NEW_STATE'] = np.where((selected_model_data['COUNTRY'] == 'United States') & (selected_model_data['STATE'] == 999) , 'United States', selected_model_data['STATE'])
selected_model_data['NEW_STATE'] = np.where(selected_model_data['COUNTRY'] != 'United States', selected_model_data['COUNTRY'], selected_model_data['NEW_STATE'])
selected_model_data['NEW_STATE'] = np.where(selected_model_data['NEW_STATE']=='ny', 'NY', selected_model_data['NEW_STATE'])
selected_model_data['NEW_STATE'] = np.where(selected_model_data['NEW_STATE']=='ca', 'CA', selected_model_data['NEW_STATE'])

test_data['NEW_STATE'] = np.where((test_data['COUNTRY'] == 'United States') & (test_data['STATE'] == 999) , 'United States', test_data['STATE'])
test_data['NEW_STATE'] = np.where(test_data['COUNTRY'] != 'United States', test_data['COUNTRY'], test_data['NEW_STATE'])
test_data['NEW_STATE'] = np.where(test_data['NEW_STATE']=='ny', 'NY', test_data['NEW_STATE'])
test_data['NEW_STATE'] = np.where(test_data['NEW_STATE']=='ca', 'CA', test_data['NEW_STATE'])

train_data = selected_model_data

train_data = train_data[['STUDENTID','STATE', 'NEW_STATE', 'COUNTRY',
       'GENDER', 'FLAG_APP', 'Days_Difference',
       'DATE_APP', 'FLAG_INTERNATIONAL', 'Citizenship',
       'FIRSTSOURCEAPP', 'MAJOR1', 'DEPT1', 'HSCODE',
       'Ugrd_Recruit_Ipeds_Ethnicity', 'CA_VISIT', 'CA_VISIT_DATE', 'ISOURCE_GRP','Ext Org Name','SUdistance']]
test_data = test_data[['STUDENTID','STATE', 'NEW_STATE','COUNTRY',
       'GENDER', 'Days_Difference',
       'FLAG_INTERNATIONAL', 'Citizenship',
       'FIRSTSOURCEAPP', 'MAJOR1', 'DEPT1', 'HSCODE',
       'Ugrd_Recruit_Ipeds_Ethnicity', 'CA_VISIT', 'CA_VISIT_DATE', 'ISOURCE_GRP','Ext Org Name','SUdistance', 'FLAG_APP']]

columns = ['NEW_STATE', 'GENDER', 'FLAG_INTERNATIONAL', 'Citizenship', 'MAJOR1', 'DEPT1', 'HSCODE',
           'Ugrd_Recruit_Ipeds_Ethnicity', 'CA_VISIT', 'ISOURCE_GRP']

new_train_data = train_data

for column in columns:
    # group data by various columns to get distribution of FLAG_APP
    grouped_train_data = train_data.groupby([column]).FLAG_APP.apply(lambda x: x.value_counts(sort=False)).reset_index()

    # Create rows artificially when count of Y = 0
    For_YES = pd.DataFrame(grouped_train_data[column])
    For_YES['level_1'] = 'Y'
    For_YES['FLAG_APP'] = 0

    # Aggregate rows to combine counts of Y
    grouped_train_data = grouped_train_data.append(For_YES)
    grouped_train_data = grouped_train_data.groupby([column, 'level_1'], as_index=False)['FLAG_APP'].sum()

    # Find total count of various categories in the categorical columns
    colname_sum = 'Sum_' + column
    grouped_train_data[colname_sum] = grouped_train_data.groupby([column], as_index=False)['FLAG_APP'].transform('sum')

    # Create new column which finds percentages of Y and N per category
    colname_new = 'New_' + column
    grouped_train_data[colname_new] = grouped_train_data['FLAG_APP'] / grouped_train_data[colname_sum]

    # Keep percentages for FLAG_APP = Y and remove the rest
    grouped_train_data = grouped_train_data[grouped_train_data.level_1 == 'Y']

    # Keep only categorical column (old and new)
    if column in ('MAJOR1', 'NEW_STATE', 'HSCODE', 'Ugrd_Recruit_Ipeds_Ethnicity', 'Citizenship'):
        grouped_train_data = grouped_train_data[[column, colname_new, colname_sum]]
    else:
        grouped_train_data = grouped_train_data[[column, colname_new]]

    # Add new modified column to the dataset
    new_train_data = pd.merge(new_train_data, grouped_train_data, how='left', on=[column])

new_train_data['New_MAJOR1'] = np.where(new_train_data['Sum_MAJOR1'] < 20, new_train_data['New_DEPT1'], new_train_data['New_MAJOR1'])
new_train_data['New_NEW_STATE'] = np.where(new_train_data['Sum_NEW_STATE']<20,
                                           new_train_data[new_train_data['NEW_STATE']==999].New_NEW_STATE.unique(),
                                           new_train_data['New_NEW_STATE'])
new_train_data['HSCODE'] = np.where(new_train_data['Sum_HSCODE'] < 20, 999, new_train_data['HSCODE'])
new_train_data['Citizenship'] = np.where(new_train_data['Sum_Citizenship'] < 20, 999, new_train_data['Citizenship'])
new_train_data['Ugrd_Recruit_Ipeds_Ethnicity'] = np.where(new_train_data['Sum_Ugrd_Recruit_Ipeds_Ethnicity'] < 20, 999, new_train_data['Ugrd_Recruit_Ipeds_Ethnicity'])

new_train_data = new_train_data.drop(['New_HSCODE', 'Sum_HSCODE'], axis=1)
new_train_data = new_train_data.drop(['New_Ugrd_Recruit_Ipeds_Ethnicity', 'Sum_Ugrd_Recruit_Ipeds_Ethnicity'], axis=1)

new_train_data = new_train_data.drop(['New_MAJOR1', 'Sum_MAJOR1'], axis=1)
new_train_data = new_train_data.drop(['New_NEW_STATE', 'Sum_NEW_STATE'], axis=1)
new_train_data = new_train_data.drop(['New_Citizenship', 'Sum_Citizenship'], axis=1)

columns = ['MAJOR1', 'NEW_STATE', 'HSCODE', 'Ugrd_Recruit_Ipeds_Ethnicity', 'Citizenship']
for column in columns:
    grouped_train_data = new_train_data.groupby(column).FLAG_APP.apply(
        lambda x: x.value_counts(sort=False)).reset_index()

    # Create rows artificially when count of Y = 0
    For_YES = pd.DataFrame(grouped_train_data[column])
    For_YES['level_1'] = 'Y'
    For_YES['FLAG_APP'] = 0

    # Aggregate rows to combine counts of Y
    grouped_train_data = grouped_train_data.append(For_YES)
    grouped_train_data = grouped_train_data.groupby([column, 'level_1'], as_index=False)['FLAG_APP'].sum()

    # Find total count of various categories in the categorical columns
    colname_sum = 'Sum_' + column
    grouped_train_data[colname_sum] = grouped_train_data.groupby([column], as_index=False)['FLAG_APP'].transform('sum')

    # Create new column which finds percentages of Y and N per category
    colname_new = 'New_' + column
    grouped_train_data[colname_new] = grouped_train_data['FLAG_APP'] / grouped_train_data[colname_sum]

    # Keep percentages for FLAG_APP = Y and remove the rest
    grouped_train_data = grouped_train_data[grouped_train_data.level_1 == 'Y']

    # Keep only categorical column (old and new)
    grouped_train_data = grouped_train_data[[column, colname_new, colname_sum]]

    # Add new modified column to the dataset
    new_train_data = pd.merge(new_train_data, grouped_train_data, how='left', on=[column])
# new_train_data[new_train_data['Sum_HSCODE']<20]

Gender = new_train_data[['GENDER','New_GENDER']].drop_duplicates()
STATE = new_train_data[['NEW_STATE','New_NEW_STATE']].drop_duplicates()
MAJOR1 = new_train_data[['MAJOR1','New_MAJOR1']].drop_duplicates(subset='MAJOR1',keep='first')
FI = new_train_data[['FLAG_INTERNATIONAL','New_FLAG_INTERNATIONAL']].drop_duplicates()
DEPT = new_train_data[['DEPT1','New_DEPT1']].drop_duplicates()
HSCODE = new_train_data[['HSCODE','New_HSCODE']].drop_duplicates()
ETHNIC = new_train_data[['Ugrd_Recruit_Ipeds_Ethnicity','New_Ugrd_Recruit_Ipeds_Ethnicity']].drop_duplicates()
CA = new_train_data[['CA_VISIT','New_CA_VISIT']].drop_duplicates()
ISOURCE = new_train_data[['ISOURCE_GRP','New_ISOURCE_GRP']].drop_duplicates()
CITIZENSHIP = new_train_data[['Citizenship','New_Citizenship']].drop_duplicates()

original_test_data = test_data[['HSCODE', 'Ugrd_Recruit_Ipeds_Ethnicity','MAJOR1','NEW_STATE', 'ISOURCE_GRP', 'Citizenship', 'DEPT1']]

columns = ['HSCODE', 'Ugrd_Recruit_Ipeds_Ethnicity','MAJOR1','NEW_STATE', 'ISOURCE_GRP', 'Citizenship', 'DEPT1']
for column in columns:
        test_data[column] = np.where(test_data[column].isin(new_train_data[column].unique()),test_data[column],999)

        Gender_NEW = pd.merge(test_data, Gender, how='left', on=['GENDER'])
        State_NEW = pd.merge(Gender_NEW, STATE, how='left', on=['NEW_STATE'])
        Major_NEW = pd.merge(State_NEW, MAJOR1, how='left', on=['MAJOR1'])
        FI_NEW = pd.merge(Major_NEW, FI, how='left', on=['FLAG_INTERNATIONAL'])
        DEPT_NEW = pd.merge(FI_NEW, DEPT, how='left', on=['DEPT1'])
        HSCODE_NEW = pd.merge(DEPT_NEW, HSCODE, how='left', on=['HSCODE'])
        ETHNIC_NEW = pd.merge(HSCODE_NEW, ETHNIC, how='left', on=['Ugrd_Recruit_Ipeds_Ethnicity'])
        CA_NEW = pd.merge(ETHNIC_NEW, CA, how='left', on=['CA_VISIT'])
        Citizenship_NEW = pd.merge(CA_NEW, CITIZENSHIP, how='left', on=['Citizenship'])
        New_test_data = pd.merge(Citizenship_NEW, ISOURCE, how='left', on=['ISOURCE_GRP'])

        new_train_data[['SUdistance']] = new_train_data[['SUdistance']].fillna(value=6000)
        New_test_data[['SUdistance']] = New_test_data[['SUdistance']].fillna(value=6000)
        New_test_data[['New_DEPT1']] = New_test_data[['New_DEPT1']].fillna(value=new_train_data['New_DEPT1'].mean())
        # New_test_data.columns[New_test_data.isna().any()].tolist()
        # New_test_data[New_test_data['New_DEPT1'].isna()]['DEPT1']
        # new_train_data[new_train_data['DEPT1']=='999']
        New_test_data = New_test_data.dropna()
        New_test_data = New_test_data.reset_index(drop=True)

        train_visit_rate = new_train_data[new_train_data['CA_VISIT'] == 1].shape[0] / new_train_data.shape[0]
        test_visit_rate = New_test_data[New_test_data['CA_VISIT'] == 1].shape[0] / New_test_data.shape[0]
        diff_visit_rate = train_visit_rate - test_visit_rate

        if diff_visit_rate > 0:
            new_encoded_value = diff_visit_rate * new_train_data[
                new_train_data['CA_VISIT'] == 1].New_CA_VISIT.unique() + \
                                (1 - diff_visit_rate) * new_train_data[
                                    new_train_data['CA_VISIT'] == 0].New_CA_VISIT.unique()
        else:
            new_encoded_value = new_train_data[new_train_data['CA_VISIT'] == 0].New_CA_VISIT.unique()

        New_test_data['New_CA_VISIT'] = np.where(New_test_data['CA_VISIT'] == 0, new_encoded_value,
                                                 New_test_data['New_CA_VISIT'])
        new_train_data['NEW_FLAG_APP'] = new_train_data.FLAG_APP.eq('Y').mul(1)

        X = ['New_NEW_STATE', 'SUdistance', 'New_Citizenship',
             'New_GENDER', 'New_FLAG_INTERNATIONAL', 'New_MAJOR1',
             'New_DEPT1', 'New_CA_VISIT', 'New_Ugrd_Recruit_Ipeds_Ethnicity',
             'New_ISOURCE_GRP', 'New_HSCODE', 'Days_Difference']
        Y = 'NEW_FLAG_APP'

        training_features = np.array(new_train_data[X])
        training_labels = new_train_data[Y].values.ravel()
        testing_features = np.array(New_test_data[X])

        split_train_data = new_train_data.sample(frac=0.5, random_state=10)
        validation_data = new_train_data[~new_train_data.index.isin(split_train_data.index)]
        split_training_features = np.array(split_train_data[X])
        split_training_labels = split_train_data[Y].values.ravel()
        validation_features = np.array(validation_data[X])
        validation_labels = validation_data[Y].values.ravel()

        New_test_data.rename(columns={'HSCODE': 'Temp_HSCODE',
                                      'Ugrd_Recruit_Ipeds_Ethnicity': 'Temp_New_Ugrd_Recruit_Ipeds_Ethnicity',
                                      'MAJOR1': 'Temp_MAJOR1',
                                      'NEW_STATE': 'Temp_NEW_STATE',
                                      'ISOURCE_GRP': 'Temp_ISOURCE_GRP',
                                      'DEPT1': 'Temp_DEPT1',
                                      'Citizenship': 'Temp_Citizenship'},
                             inplace=True)
        New_test_data = New_test_data.join(original_test_data)
        New_test_data = New_test_data[['STUDENTID', 'STATE', 'NEW_STATE', 'COUNTRY', 'GENDER',
                                       'Days_Difference', 'FLAG_INTERNATIONAL', 'Citizenship',
                                       'FIRSTSOURCEAPP', 'MAJOR1', 'DEPT1', 'HSCODE',
                                       'Ugrd_Recruit_Ipeds_Ethnicity', 'CA_VISIT', 'CA_VISIT_DATE',
                                       'ISOURCE_GRP', 'Ext Org Name', 'SUdistance', 'FLAG_APP', 'New_GENDER',
                                       'Temp_NEW_STATE', 'New_NEW_STATE', 'Temp_MAJOR1', 'New_MAJOR1',
                                       'New_FLAG_INTERNATIONAL', 'Temp_DEPT1', 'New_DEPT1',
                                       'Temp_HSCODE', 'New_HSCODE', 'Temp_New_Ugrd_Recruit_Ipeds_Ethnicity',
                                       'New_Ugrd_Recruit_Ipeds_Ethnicity', 'New_CA_VISIT',
                                       'Temp_Citizenship', 'New_Citizenship', 'New_ISOURCE_GRP']]



        scaler = StandardScaler()
        scaler.fit(training_features)
        standardized_training = scaler.transform(training_features)

        scaler.fit(testing_features)
        standardized_testing = scaler.transform(testing_features)

        scaler.fit(split_training_features)
        standardized_split_training = scaler.transform(split_training_features)

        scaler.fit(validation_features)
        standardized_validation = scaler.transform(validation_features)

        from sklearn.model_selection import GridSearchCV

        from sklearn.linear_model import LogisticRegression

        #p_test = {'solver': ['lbfgs'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [500, 1000]}
       # lr_tuning = GridSearchCV(estimator=LogisticRegression(random_state=0),
        #                         param_grid=p_test, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
      #  lr_tuning.fit(training_features, training_labels)



        import pickle

        #pickle.dump(lr_tuning, open("C:/Users/chandiramania/Desktop/Data/lr_model.pkl", "wb"))



        loaded_model = pickle.load(open("C:/Users/chandiramania/Desktop/Data/model.pkl", "rb"))
        loaded_model_lr = pickle.load(open("C:/Users/chandiramania/Desktop/Data/lr_model.pkl", "rb"))
        result = loaded_model.predict_proba(testing_features)[:,1]
        result = pd.DataFrame({"gbm_predictions": result})

        result_lr = loaded_model_lr.predict_proba(testing_features)[:, 1]
        result_lr = pd.DataFrame({"lr_predictions": result_lr})

        #result = New_test_data.join(result1)

        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve

        gbm_roc_auc = roc_auc_score(training_labels, loaded_model.predict_proba(training_features)[:,1])
        fpr, tpr, thresholds = roc_curve(training_labels, loaded_model.predict_proba(training_features)[:,1])

        gbm_roc_auc_lr = roc_auc_score(training_labels, loaded_model_lr.predict_proba(training_features)[:, 1])
        fpr_lr, tpr_lr, thresholds_lr = roc_curve(training_labels, loaded_model_lr.predict_proba(training_features)[:, 1])


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
df = pd.read_csv(DATA_PATH.joinpath("clinical_analytics.csv"))

clinic_list = df["Clinic Name"].unique()
df["Admit Source"] = df["Admit Source"].fillna("Not Identified")
admit_list = df["Admit Source"].unique().tolist()

# Date
# Format checkin Time
df["Check-In Time"] = df["Check-In Time"].apply(
    lambda x: dt.strptime(x, "%Y-%m-%d %I:%M:%S %p")
)  # String -> Datetime

# Insert weekday and hour of checkin time
df["Days of Wk"] = df["Check-In Hour"] = df["Check-In Time"]
df["Days of Wk"] = df["Days of Wk"].apply(
    lambda x: dt.strftime(x, "%A")
)  # Datetime -> weekday string

df["Check-In Hour"] = df["Check-In Hour"].apply(
    lambda x: dt.strftime(x, "%I %p")
)  # Datetime -> int(hour) + AM/PM

day_list = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

check_in_duration = df["Check-In Time"].describe()

# Register all departments for callbacks
all_departments = df["Department"].unique().tolist()
wait_time_inputs = [
    Input((i + "_wait_time_graph"), "selectedData") for i in all_departments
]
score_inputs = [Input((i + "_score_graph"), "selectedData") for i in all_departments]


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("App Modeling"),html.Div(
                id="intro",
                children="Explore various models.",
            ),
        ],
    )


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Model"),
            dcc.Dropdown(
                id="model-select",
                options=[{"label": i, "value": i} for i in ['GBM', 'LR']],
                value=clinic_list[0],
            ),
            html.Br(),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
        ],
    )


def generate_patient_volume_heatmap(clinic, hm_click, admit_type, reset):
    """
    :param: start: start date from selection.
    :param: end: end date from selection.
    :param: clinic: clinic from selection.
    :param: hm_click: clickData from heatmap.
    :param: admit_type: admission type from selection.
    :param: reset (boolean): reset heatmap graph if True.
    :return: Patient volume annotated heatmap.
    """



    #x_axis = [datetime.time(i).strftime("%I %p") for i in range(24)]  # 24hr time list
    #y_axis = day_list

    x_axis = fpr
    y_axis = tpr



    data = [
        dict(
            x=x_axis,
            y=y_axis,
            name="",
           )
    ]

    layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )
    return {"data": data, "layout": layout}


def generate_table_row(id, style, col1, col2, col3):
    """ Generate table rows.
    :param id: The ID of table row.
    :param style: Css style of this row.
    :param col1 (dict): Defining id and children for the first column.
    :param col2 (dict): Defining id and children for the second column.
    :param col3 (dict): Defining id and children for the third column.
    """

    return html.Div(
        id=id,
        className="row table-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                style={"display": "table", "height": "100%"},
                className="two columns row-department",
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "height": "100%"},
                className="five columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"textAlign": "center", "height": "100%"},
                className="five columns",
                children=col3["children"],
            ),
        ],
    )


def generate_table(model, max_rows=10):
    if model == 'LR':
        return html.Table(
        # Header
            [html.Tr([html.Th(col) for col in result_lr.columns])] +

        # Body
            [html.Tr([
             html.Td(result_lr.iloc[i][col]) for col in result_lr.columns]) for i in range(min(len(result_lr), max_rows))]
         )
    else:
        return html.Table(
            # Header
            [html.Tr([html.Th(col) for col in result.columns])] +

            # Body
            [html.Tr([
                html.Td(result.iloc[i][col]) for col in result.columns]) for i in
                range(min(len(result), max_rows))]
        )

def generate_table_row_helper(department):
    """Helper function.
    :param: department (string): Name of department.
    :return: Table row.
    """
    return generate_table_row(
        department,
        {},
        {"id": department + "_department", "children": html.B(department)},
        {
            "id": department + "wait_time",
            "children": dcc.Graph(
                id=department + "_wait_time_graph",
                style={"height": "100%", "width": "100%"},
                className="wait_time_graph",
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure={
                    "layout": dict(
                        margin=dict(l=0, r=0, b=0, t=0, pad=0),
                        xaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                },
            ),
        },
        {
            "id": department + "_patient_score",
            "children": dcc.Graph(
                id=department + "_score_graph",
                style={"height": "100%", "width": "100%"},
                className="patient_score_graph",
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure={
                    "layout": dict(
                        margin=dict(l=0, r=0, b=0, t=0, pad=0),
                        xaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showline=False,
                            showticklabels=False,
                            zeroline=False,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                },
            ),
        },
    )


def initialize_table():
    """
    :return: empty table children. This is intialized for registering all figure ID at page load.
    """

    # header_row
    header = [
        generate_table_row(
            "header",
            {"height": "50px"},
            {"id": "header_department", "children": html.B("Department")},
            {"id": "header_wait_time_min", "children": html.B("Wait Time Minutes")},
            {"id": "header_care_score", "children": html.B("Care Score")},
        )
    ]

    # department_row
    rows = [generate_table_row_helper(department) for department in all_departments]
    header.extend(rows)
    empty_table = header

    return empty_table


def generate_patient_table(figure_list, departments, wait_time_xrange, score_xrange):
    """
    :param score_xrange: score plot xrange [min, max].
    :param wait_time_xrange: wait time plot xrange [min, max].
    :param figure_list:  A list of figures from current selected metrix.
    :param departments:  List of departments for making table.
    :return: Patient table.
    """
    # header_row
    header = [
        generate_table_row(
            "header",
            {"height": "50px"},
            {"id": "header_department", "children": html.B("Department")},
            {"id": "header_wait_time_min", "children": html.B("Wait Time Minutes")},
            {"id": "header_care_score", "children": html.B("Care Score")},
        )
    ]

    # department_row
    rows = [generate_table_row_helper(department) for department in departments]
    # empty_row
    empty_departments = [item for item in all_departments if item not in departments]
    empty_rows = [
        generate_table_row_helper(department) for department in empty_departments
    ]

    # fill figures into row contents and hide empty rows
    for ind, department in enumerate(departments):
        rows[ind].children[1].children.figure = figure_list[ind]
        rows[ind].children[2].children.figure = figure_list[ind + len(departments)]
    for row in empty_rows[1:]:
        row.style = {"display": "none"}

    # convert empty row[0] to axis row
    empty_rows[0].children[0].children = html.B(
        "graph_ax", style={"visibility": "hidden"}
    )

    empty_rows[0].children[1].children.figure["layout"].update(
        dict(margin=dict(t=-70, b=50, l=0, r=0, pad=0))
    )

    empty_rows[0].children[1].children.config["staticPlot"] = True

    empty_rows[0].children[1].children.figure["layout"]["xaxis"].update(
        dict(
            showline=True,
            showticklabels=True,
            tick0=0,
            dtick=20,
            range=wait_time_xrange,
        )
    )
    empty_rows[0].children[2].children.figure["layout"].update(
        dict(margin=dict(t=-70, b=50, l=0, r=0, pad=0))
    )

    empty_rows[0].children[2].children.config["staticPlot"] = True

    empty_rows[0].children[2].children.figure["layout"]["xaxis"].update(
        dict(showline=True, showticklabels=True, tick0=0, dtick=0.5, range=score_xrange)
    )

    header.extend(rows)
    header.extend(empty_rows)
    return header





def create_table_figure(
    department, filtered_df, category, category_xrange, selected_index
):
    """Create figures.
    :param department: Name of department.
    :param filtered_df: Filtered dataframe.
    :param category: Defining category of figure, either 'wait time' or 'care score'.
    :param category_xrange: x axis range for this figure.
    :param selected_index: selected point index.
    :return: Plotly figure dictionary.
    """
    aggregation = {
        "Wait Time Min": "mean",
        "Care Score": "mean",
        "Days of Wk": "first",
        "Check-In Time": "first",
        "Check-In Hour": "first",
    }

    df_by_department = filtered_df[
        filtered_df["Department"] == department
    ].reset_index()
    grouped = (
        df_by_department.groupby("Encounter Number").agg(aggregation).reset_index()
    )
    patient_id_list = grouped["Encounter Number"]

    x = grouped[category]
    y = list(department for _ in range(len(x)))

    f = lambda x_val: dt.strftime(x_val, "%Y-%m-%d")
    check_in = (
        grouped["Check-In Time"].apply(f)
        + " "
        + grouped["Days of Wk"]
        + " "
        + grouped["Check-In Hour"].map(str)
    )

    text_wait_time = (
        "Patient # : "
        + patient_id_list
        + "<br>Check-in Time: "
        + check_in
        + "<br>Wait Time: "
        + grouped["Wait Time Min"].round(decimals=1).map(str)
        + " Minutes,  Care Score : "
        + grouped["Care Score"].round(decimals=1).map(str)
    )

    layout = dict(
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        clickmode="event+select",
        hovermode="closest",
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            range=category_xrange,
        ),
        yaxis=dict(
            showgrid=False, showline=False, showticklabels=False, zeroline=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    trace = dict(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=14, line=dict(width=1, color="#ffffff")),
        color="#2c82ff",
        selected=dict(marker=dict(color="#ff6347", opacity=1)),
        unselected=dict(marker=dict(opacity=0.1)),
        selectedpoints=selected_index,
        hoverinfo="text",
        customdata=patient_id_list,
        text=text_wait_time,
    )

    return {"data": [trace], "layout": layout}


app.layout = html.Div(
    [
    html.H1('Dash Tabs component demo'),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Tab One', value='tab-1-example'),
        dcc.Tab(label='Tab Two', value='tab-2-example'),
    ]),
    html.Div(
    id="app-container",
    children=[
        # Banner

        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Patient Volume Heatmap
                html.Div(
                    dcc.Graph(id='indicator-graphic')
                ),
                # Patient Wait time by Department
                html.Div(
                    id="wait_time_card",
                    children=[
                        html.B("Patient Wait Time and Satisfactory Scores"),
                        html.Hr(),
                        generate_table('LR'),
                        #html.Div(id="wait_time_table", children=generate_table('LR')),
                    ],
                ),
            ],
        ),
    ],
    ),
    ],
    )


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("wait_time_table", "children")] + wait_time_inputs + score_inputs,
)


@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return html.Div([
            html.H3('Tab content 1'),
            dcc.Graph(
                id='graph-1-tabs',
                figure={
                    'data': [{
                        'x': [1, 2, 3],
                        'y': [3, 1, 2],
                        'type': 'bar'
                    }]
                }
            )
        ])
    elif tab == 'tab-2-example':
        return html.Div([
            html.H3('Tab content 2'),
            dcc.Graph(
                id='graph-2-tabs',
                figure={
                    'data': [{
                        'x': [1, 2, 3],
                        'y': [5, 10, 6],
                        'type': 'bar'
                    }]
                }
            )
        ])

@app.callback(

    Output('indicator-graphic', 'figure'),
    [

        Input("model-select", "value"),

        Input("reset-btn", "n_clicks"),
    ]
    
)
def update_graph(model, reset):

    if model == 'LR':
        return {
            'data': [dict(
                x=fpr_lr,
                # dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                y=tpr_lr,
                # dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                text='Heading',
                # dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                mode='line',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
            'layout': dict(
                xaxis={
                    'title': 'False positive rate',
                    'type': 'linear'
                },
                yaxis={
                    'title': 'True positive rate',
                    'type': 'linear'
                },
                margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                hovermode='closest'
            )
        }
    else:
        return {
            'data': [dict(
                x=fpr,
                # dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                y=tpr,
                label='GBM (area = %0f)' % gbm_roc_auc_lr,
                # dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                text='Heading',
                # dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
                mode='line',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
            'layout': dict(
                xaxis={
                    'title': 'False positive rate',
                    'type': 'linear'
                },
                yaxis={
                    'title': 'True positive rate',
                    'type': 'linear'
                },

                margin={'l': 60, 'b': 40, 't': 40, 'r': 40},
                hovermode='closest'
            )
        }


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)