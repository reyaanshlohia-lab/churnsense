"""ChurnSense: the analysis engine, running in the browser (Pyodide).

Built from the same code as people_prediction_app.ipynb. Differences in the browser: one process (no parallel
search), and explanations without the shap package (tree path contributions for tree models, permutation SHAP for
the others).
"""

import os
import re
import io
import glob
import copy
import math
import shutil
import zipfile
import tempfile
import warnings
import functools
import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterSampler, check_cv
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.neighbors._base import _get_weights   # KNN's own neighbour weighting
from sklearn.metrics import get_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, mean_absolute_error, r2_score
)

# display() only exists inside Jupyter/Colab; fall back to print when run as a script
try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = str

# google.colab only exists in Colab; the notebook also runs locally without it
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

pio.templates.default = 'plotly_white'
if IN_COLAB:
    pio.renderers.default = 'colab'
    # PNG chart downloads need Chrome for Kaleido; Colab does not have it, so download it once (about 30 s)
    try:
        import kaleido
        kaleido.get_chrome_sync()
    except Exception as error:
        print('PNG export will not be available (charts download as HTML instead):', error)

OUTPUT_DIR = tempfile.mkdtemp(prefix='people_prediction_')  # downloads are written here


SEARCH_PRESETS = {
    'Quick': {
        'Logistic Regression': {'cv': 5, 'n_iter': 6, 'grid': {'C': [0.01, 0.1, 1, 10, 100, 1000]}},
        'Decision Tree': {'cv': 5, 'n_iter': 8, 'grid': {'max_depth': [3, 5, 8, None], 'min_samples_leaf': [1, 5, 10, 20]}},
        'K-Nearest Neighbors': {'cv': 5, 'n_iter': 10,
                                'grid': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance'], 'p': [1, 2]}},
        'Random Forest': {'cv': 3, 'n_iter': 5,
                          'grid': {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_leaf': [1, 5]}},
        'Neural Network': {'cv': 3, 'n_iter': 4, 'grid': {'hidden_layer_sizes': [(8,), (16,), (32,), (64,)]}},
    },
    # Thorough: wide enough that, on every sample dataset, no best value is at the edge of the values tested
    # (n_iter keeps roughly the same share of combinations as the Quick preset)
    'Thorough': {
        'Logistic Regression': {'cv': 5, 'n_iter': 8, 'grid': {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}},
        'Decision Tree': {'cv': 5, 'n_iter': 24,
                          'grid': {'max_depth': [1, 2, 3, 5, 8, None],
                                   'min_samples_leaf': [1, 5, 10, 20, 50, 100, 200, 400]}},
        'K-Nearest Neighbors': {'cv': 5, 'n_iter': 25,
                                'grid': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 21, 31, 51, 75, 101, 151, 201, 301, 501,
                                                         751, 1001],
                                         'weights': ['uniform', 'distance'], 'p': [1, 2]}},
        'Random Forest': {'cv': 3, 'n_iter': 42,
                          'grid': {'n_estimators': [100, 200, 400, 800, 1000], 'max_depth': [3, 5, 10, None],
                                   'min_samples_leaf': [1, 5, 10, 20, 50]}},
        'Neural Network': {'cv': 3, 'n_iter': 12,
                           'grid': {'hidden_layer_sizes': [(4,), (8,), (16,), (32,), (64,), (128,), (256,), (512,),
                                                           (1024,), (2048,), (4096,), (8192,)]}},
    },
}

DEFAULT_PRESET = 'Thorough'   # the search preset the app starts with ('Quick' is about 5x faster)
# Quick chooses each model's settings on a stratified sample of at most this many training rows (big files train in
# about a third of the time); the best settings are then trained on all the training rows. Thorough uses every row.
PRESET_SEARCH_ROWS = {'Quick': 4000, 'Thorough': None}

DEFAULTS = {
    'missing_strategy': 'median',          # median / mean / mode / drop rows
    'exclude_sensitive': True,             # leave sensitive columns out of training
    'test_size': 0.2,
    'random_state': 42,
    'scoring': 'ROC-AUC',                  # ROC-AUC / F1 / Accuracy / Recall
    'threshold': 0.5,
    'level_cutoffs': (0.3, 0.6),           # Low < 30% <= Medium < 60% <= High
    'preset': DEFAULT_PRESET,
    'models': {
        name: {'run': True, **copy.deepcopy(spec)} for name, spec in SEARCH_PRESETS[DEFAULT_PRESET].items()
    },
    'logistic_max_iter': 2000,
    'balanced': True,                      # class_weight='balanced' for LR, Decision Tree and Random Forest
    'mlp_max_iter': 1000,                  # was 300 in the original, which stopped before converging
    'clustering': {'n_clusters': 3, 'n_init': 10, 'max_k': 10},
    'regression': {'target': None, 'features': None},  # chosen per dataset
}

MIN_ROWS = 30
SCORING = {'ROC-AUC': 'roc_auc', 'F1': 'f1', 'Accuracy': 'accuracy', 'Recall': 'recall'}
MISSING_STRATEGIES = ['median', 'mean', 'mode', 'drop rows']
MODEL_NAMES = list(SEARCH_PRESETS['Quick'])

# words that mark a column as sensitive (matched against the words in a column name)
SENSITIVE_WORDS = {'gender', 'sex', 'age', 'race', 'ethnicity', 'ethnic', 'religion', 'religious', 'disability',
                   'disabled', 'marital', 'married', 'nationality', 'senior', 'citizenship'}

RULE_COLUMNS = ['Feature', 'Condition', 'Value', 'Suggestion']
CONDITIONS = ['<', '<=', '>', '>=', '==', '!=', 'any']


def rules_table(rows):
    return pd.DataFrame(rows, columns=RULE_COLUMNS)


SCENARIOS = {
    'Employees': {
        'name': 'Employees leaving a business (employee attrition)',
        'mode': 'leave', 'person': 'employee', 'people': 'employees', 'org': 'company',
        'outcome_past': 'left', 'outcome_ing': 'leaving', 'stay_past': 'stayed',
        'sample': 'employees_SYNTHETIC.csv',
        'targets': ['Attrition', 'Left', 'Turnover', 'Quit'],
        'id_columns': ['EmployeeNumber', 'EmployeeID', 'EmployeeId', 'ID'],
        'sensitive': ['Age', 'Gender', 'MaritalStatus'],
        'regression': ('MonthlyIncome', ['JobLevel', 'TotalWorkingYears', 'YearsAtCompany']),
        'rules': rules_table([
            ['OverTime', '==', 'Yes', 'Review workload and overtime; consider extra staff or time off in lieu.'],
            ['YearsSinceLastPromotion', '>=', '3', 'Discuss a career path and a promotion review.'],
            ['JobSatisfaction', '<=', '2', 'Hold a stay interview about what would make the job more satisfying.'],
            ['EnvironmentSatisfaction', '<=', '2', 'Check the team environment with the manager.'],
            ['WorkLifeBalance', '<=', '2', 'Offer flexible hours or remote days.'],
            ['MonthlyIncome', '<', '3000', 'Benchmark pay against the market for this role.'],
            ['StockOptionLevel', '==', '0', 'Consider long-term incentives such as stock options.'],
            ['BusinessTravel', '==', 'Travel_Frequently', 'Look at reducing business travel.'],
            ['DistanceFromHome', '>=', '15', 'Consider hybrid work to cut commuting.'],
            ['YearsAtCompany', '<', '2', 'Strengthen onboarding and assign a mentor.'],
            ['EngagementScore', '<=', '4', 'Hold an engagement check-in with the manager.'],
            ['RecruiterContactsLastYear', '>=', '4', 'Hold a stay interview: they are being approached by recruiters.'],
            ['SalaryVsMarketPct', '<', '95', 'Bring pay closer to the market rate.'],
            ['CareerPathDiscussed', '==', 'No', 'Agree a career path and next steps.'],
        ]),
    },
    'Customers': {
        'name': 'Clients/customers leaving a business (customer churn)',
        'mode': 'leave', 'person': 'customer', 'people': 'customers', 'org': 'company',
        'outcome_past': 'churned', 'outcome_ing': 'churning', 'stay_past': 'stayed',
        'sample': 'customers_SYNTHETIC.csv',
        'targets': ['Churn', 'Churned', 'Exited', 'Left'],
        'id_columns': ['customerID', 'CustomerID', 'CustomerId', 'ID'],
        'sensitive': ['gender', 'SeniorCitizen', 'Age'],
        'regression': ('MonthlyCharges', ['SeniorCitizen', 'tenure', 'TotalCharges']),   # as in the original notebook
        'rules': rules_table([
            ['Contract', '==', 'Month-to-month', 'Offer a discount for moving to a 1- or 2-year contract.'],
            ['tenure', '<', '12', 'Strengthen onboarding and check in during the first year.'],
            ['TechSupport', '==', 'No', 'Offer a free tech support trial.'],
            ['OnlineSecurity', '==', 'No', 'Offer a free online security trial.'],
            ['PaymentMethod', '==', 'Electronic check', 'Encourage automatic payment (card or bank transfer).'],
            ['MonthlyCharges', '>', '80', 'Review the plan; suggest a cheaper bundle.'],
            ['InternetService', '==', 'Fiber optic', 'Check fiber service quality and price for this customer.'],
            ['PaperlessBilling', '==', 'Yes', 'Make sure bills are clear and easy to understand.'],
            ['SatisfactionScore', '<=', '4', 'Call the customer to understand and fix their problems.'],
            ['CompetitorOfferReceived', '==', 'Yes', 'Offer a matching retention deal.'],
            ['NetworkOutagesLastQuarter', '>=', '2', 'Check the network in their area and offer a service credit.'],
            ['SupportTicketsLastYear', '>=', '3', 'Assign a dedicated support agent.'],
        ]),
    },
    'Students': {
        'name': 'Students leaving a school (student dropout)',
        'mode': 'leave', 'person': 'student', 'people': 'students', 'org': 'school',
        'outcome_past': 'dropped out', 'outcome_ing': 'dropping out', 'stay_past': 'stayed in school',
        'sample': 'students_SYNTHETIC.csv',
        'targets': ['DroppedOut', 'Dropout', 'Target', 'Left'],
        'id_columns': ['StudentID', 'StudentId', 'ID'],
        'sensitive': ['Gender', 'Age'],
        'regression': ('AverageGrade', ['AttendanceRate', 'FailedSubjects', 'DisciplinaryIncidents']),
        'rules': rules_table([
            ['AttendanceRate', '<', '0.85', 'Contact the family about attendance and agree an attendance plan.'],
            ['FailedSubjects', '>=', '2', 'Arrange tutoring for the failed subjects.'],
            ['FeePaymentStatus', '!=', 'Paid', 'Discuss a payment plan or the hardship fund.'],
            ['DisciplinaryIncidents', '>=', '2', 'Assign a mentor and behaviour support.'],
            ['GradeTrend', '==', 'Declining', 'Hold an early academic check-in.'],
            ['Extracurricular', '==', 'No', 'Encourage joining a club or activity.'],
            ['ParentalInvolvement', '==', 'Low', 'Invite the parents to a meeting.'],
            ['CounsellingSessions', '==', '0', 'Offer a counselling session.'],
            ['Scholarship', '==', 'No', 'Check eligibility for financial support.'],
            ['SchoolOffersCareerPath', '==', 'No', 'Talk about their career goal: explain stream options, partnerships or coaching the school can offer.'],
            ['OfferFromAnotherSchool', '==', 'Yes', 'Meet the family about the other offer; consider a merit scholarship or an advanced programme.'],
            ['SchoolSatisfaction', '<=', '2', 'Ask the student what would improve their experience at school.'],
            ['BullyingReported', '==', 'Yes', 'Follow the anti-bullying plan and check in every week.'],
            ['HealthIssues', '==', 'Yes', 'Work with the school nurse and plan catch-up support.'],
            ['CareerCounsellingAtSchool', '==', 'No', 'Book a career counselling session.'],
        ]),
    },
    'Teachers': {
        'name': 'Teachers leaving a school (teacher attrition)',
        'mode': 'leave', 'person': 'teacher', 'people': 'teachers', 'org': 'school',
        'outcome_past': 'left', 'outcome_ing': 'leaving', 'stay_past': 'stayed',
        'sample': 'teachers_SYNTHETIC.csv',
        'targets': ['Left', 'Attrition', 'Turnover'],
        'id_columns': ['TeacherID', 'TeacherId', 'ID'],
        'sensitive': ['Gender', 'Age'],
        'regression': ('JobSatisfaction', ['WorkloadHoursPerWeek', 'AdminSupportRating', 'StudentBehaviourRating']),
        'rules': rules_table([
            ['WorkloadHoursPerWeek', '>', '50', 'Reduce non-teaching duties and admin load.'],
            ['ClassSize', '>', '30', 'Review class sizes or add a teaching assistant.'],
            ['YearsSinceLastRaise', '>=', '3', 'Review pay progression.'],
            ['AdminSupportRating', '<=', '2', 'Improve support from school leadership.'],
            ['ProfessionalDevelopmentHours', '<', '10', 'Offer professional development opportunities.'],
            ['ContractType', '==', 'Temporary', 'Discuss a permanent contract.'],
            ['StudentBehaviourRating', '<=', '2', 'Provide behaviour management support.'],
            ['JobSatisfaction', '<=', '2', 'Hold a stay interview.'],
            ['YearsExperience', '<', '3', 'Pair new teachers with a mentor.'],
            ['OfferFromAnotherSchool', '==', 'Yes', 'Hold a stay conversation and review pay and role.'],
            ['BurnoutScore', '>=', '7', 'Reduce workload and offer wellbeing support.'],
            ['OvertimeHoursPerWeek', '>', '8', 'Review extra duties and overtime.'],
            ['CareerGrowthRating', '<=', '2', 'Discuss a growth path: leadership roles or training.'],
        ]),
    },
    'Custom': {
        'name': 'Custom: any dataset, leave or join',
        'mode': 'join', 'person': 'applicant', 'people': 'applicants', 'org': 'school',
        'outcome_past': 'enrolled', 'outcome_ing': 'enrolling', 'stay_past': 'did not enrol',
        'sample': 'custom_join_SYNTHETIC.csv',
        'targets': ['Enrolled', 'Joined', 'Converted', 'Accepted', 'Outcome', 'Target'],
        'id_columns': ['ApplicantID', 'ID'],
        'sensitive': ['Gender'],
        'regression': ('EntranceScore', ['DistanceKm', 'FollowUpContacts']),
        'rules': rules_table([
            ['ScholarshipOffered', '==', 'No', 'Consider offering a scholarship or bursary.'],
            ['CampusVisit', '==', 'No', 'Invite the applicant for a campus visit or open day.'],
            ['FollowUpContacts', '<', '2', 'Follow up by phone or email.'],
            ['DistanceKm', '>', '30', 'Explain transport or boarding options.'],
            ['FeeLevel', '==', 'High', 'Explain payment plans and financial aid.'],
            ['SiblingsEnrolled', '==', '0', 'Invite the family to meet current students and parents.'],
            ['OfferFromOtherSchool', '==', 'Yes', 'Explain what makes the school different; consider matching a scholarship.'],
            ['SchoolOffersPreferredStream', '==', 'No', 'Explain the subject options or partnerships for their preferred stream.'],
            ['HostelNeeded', '==', 'Yes', 'Confirm hostel or transport options early.'],
            ['DocumentsComplete', '==', 'No', 'Help the family complete the documents.'],
            ['ApplicationFeePaid', '==', 'No', 'Send a reminder about the application fee and deadline.'],
        ]),
    },
}

# wording when the Custom scenario is switched between Leave and Join with no other wording given
CUSTOM_WORDING = {
    'leave': {'person': 'member', 'people': 'members', 'org': 'organisation',
              'outcome_past': 'left', 'outcome_ing': 'leaving', 'stay_past': 'stayed'},
    'join': {'person': 'applicant', 'people': 'applicants', 'org': 'organisation',
             'outcome_past': 'joined', 'outcome_ing': 'joining', 'stay_past': 'did not join'},
}

# starter rules for a Custom dataset without its own rules; '*' matches any of a person's top reasons
STARTER_RULES = {
    'leave': rules_table([
        ['*', 'any', '', "Talk with them about '{feature}', one of the main reasons behind their risk of {outcome_ing}."],
        ['*', 'any', '', "Check whether something can be changed about '{feature}' to keep them."],
    ]),
    'join': rules_table([
        ['*', 'any', '', "Address '{feature}' in your outreach (for example an offer, a visit or a follow-up)."],
        ['*', 'any', '', "Follow up personally about '{feature}', which is holding back their likelihood of {outcome_ing}."],
    ]),
}


SAMPLE_FOLDERS = ['sample_data', '.', '/content', '/content/sample_data',
                  '/content/drive/MyDrive', '/content/drive/MyDrive/sample_data']
SAMPLE_URLS = {
    'customers.csv': 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
    'employees.csv': 'https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv',
}
# other names the real files may have been uploaded under
SAMPLE_ALIASES = {
    'customers.csv': ['WA_Fn-UseC_-Telco-Customer-Churn*.csv', 'Telco-Customer-Churn*.csv'],
    'employees.csv': ['WA_Fn-UseC_-HR-Employee-Attrition*.csv', 'emp_attrition*.csv'],
}


def _yes_no(mask):
    return np.where(mask, 'Yes', 'No')


def _outcome(rng, logit, flip=0.03):
    """Draw Yes/No outcomes from a logit, then flip a few labels so no model can be perfect."""
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(len(p)) < p
    flip_mask = rng.random(len(p)) < flip
    return _yes_no(np.where(flip_mask, ~y, y))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _pick(rng, options, n, p=None):
    return rng.choice(options, n, p=p)


def _scale(rng, centre, n, low=1, high=5, spread=0.9):
    """A 1-5 (or low-high) rating around a centre, with noise."""
    return np.clip(np.round(centre + rng.normal(0, spread, n)), low, high).astype(int)


def make_students(n=12000, seed=42):
    """Students who stayed or dropped out. Two kinds of leaving are built in: students who struggle (attendance,
    failed subjects, fees, little support) and HIGH ACHIEVERS who move to a school that fits their career goal
    better (no stream or coaching for their goal here, an offer from another school, low satisfaction)."""
    rng = np.random.default_rng(seed)
    grade = rng.integers(9, 13, n)
    age = grade + 5 + rng.integers(0, 2, n)
    stream = np.where(grade >= 11, _pick(rng, ['Science', 'Commerce', 'Humanities'], n, [0.45, 0.33, 0.22]), 'General')
    income = _pick(rng, ['Low', 'Middle', 'High'], n, [0.3, 0.48, 0.22])
    parent_ed = np.where(income == 'High', _pick(rng, ['School', 'Graduate', 'Postgraduate'], n, [0.1, 0.45, 0.45]),
                         np.where(income == 'Middle', _pick(rng, ['School', 'Graduate', 'Postgraduate'], n, [0.3, 0.5, 0.2]),
                                  _pick(rng, ['School', 'Graduate', 'Postgraduate'], n, [0.65, 0.3, 0.05])))
    first_gen = (parent_ed == 'School') & (rng.random(n) < 0.8)
    parents = np.where(income == 'Low', _pick(rng, ['Low', 'Medium', 'High'], n, [0.4, 0.42, 0.18]),
                       _pick(rng, ['Low', 'Medium', 'High'], n, [0.18, 0.45, 0.37]))
    boarder = rng.random(n) < 0.12
    distance = np.where(boarder, 0.5, np.round(rng.gamma(2, 3.2, n), 1))
    transport = np.where(boarder, 'Boarding', np.where(distance < 2, _pick(rng, ['Walk', 'Cycle'], n),
                         _pick(rng, ['School bus', 'Private', 'Public transport'], n, [0.5, 0.3, 0.2])))
    health = rng.random(n) < 0.09
    sick_days = rng.poisson(2 + 5 * health)
    attendance = np.clip(rng.beta(9, 1.3, n) - 0.04 * health - 0.002 * distance, 0.35, 1.0).round(3)
    late = rng.poisson(np.clip(1 + 0.25 * distance - 6 * (attendance - 0.9), 0.2, None))
    study = np.clip(np.round(rng.gamma(3, 3.2, n) + 2 * (parents == 'High')), 0, 40).astype(int)
    homework = np.clip(rng.beta(6, 1.6, n) + 0.04 * (parents == 'High') - 0.05 * (parents == 'Low'), 0.1, 1).round(2)
    ability = rng.normal(0, 1, n)                        # hidden talent
    average = np.clip(64 + 9 * ability + 30 * (attendance - 0.88) + 0.35 * (study - 9) + 14 * (homework - 0.8)
                      + rng.normal(0, 4, n), 18, 99).round(1)
    def subject(spread):
        return np.clip(np.round(average + rng.normal(0, spread, n)), 5, 100).astype(int)
    maths, science, english, social, language = subject(9), subject(8), subject(6), subject(6), subject(7)
    percentile = np.round(pd.Series(average).groupby(grade).rank(pct=True).to_numpy() * 100).astype(int)
    trend = _pick(rng, ['Improving', 'Stable', 'Declining'], n, [0.25, 0.5, 0.25])
    failed = ((maths < 35).astype(int) + (science < 35) + (english < 35) + (social < 35) + (language < 35))
    library = rng.poisson(np.clip(1.5 + 0.08 * (average - 60), 0.2, None))
    online = np.round(rng.gamma(2, 1.6, n), 1)
    incidents = rng.poisson(np.clip(0.5 - 0.01 * (average - 64), 0.05, None))
    extra = rng.random(n) < 0.45
    sports = rng.random(n) < 0.3
    olympiads = rng.poisson(np.clip(0.1 + 0.05 * (average - 70), 0.02, None))
    scholarship = rng.random(n) < np.clip(0.08 + 0.006 * (average - 64) + 0.12 * (income == 'Low'), 0.02, 0.6)
    fee = np.where(income == 'Low', _pick(rng, ['Paid', 'Late', 'Unpaid'], n, [0.55, 0.27, 0.18]),
                   _pick(rng, ['Paid', 'Late', 'Unpaid'], n, [0.85, 0.11, 0.04]))
    ptm = np.clip(rng.poisson(1.2 + 1.2 * (parents == 'High') - 0.6 * (parents == 'Low')), 0, 4)
    counselling = rng.poisson(1.0, n)
    peers = _scale(rng, 3.4, n)
    bullying = rng.random(n) < np.clip(0.06 + 0.06 * (peers <= 2), 0, 1)
    teacher_rating = _scale(rng, 3.6, n)
    goal = _pick(rng, ['Engineering', 'Medicine', 'Computer Science', 'Business', 'Law', 'Arts & Design',
                       'Civil Services', 'Undecided'], n, [0.18, 0.14, 0.12, 0.14, 0.07, 0.08, 0.07, 0.20])
    offers_path_p = np.select([np.isin(goal, ['Engineering', 'Medicine', 'Computer Science']), goal == 'Arts & Design',
                               goal == 'Law', goal == 'Undecided'], [0.6, 0.45, 0.7, 1.0], 0.85)
    offers_path = rng.random(n) < offers_path_p
    coaching = rng.random(n) < np.clip(0.08 + 0.35 * np.isin(goal, ['Engineering', 'Medicine'])
                                       + 0.012 * (average - 64), 0.02, 0.85)
    career_counselling = rng.random(n) < 0.55
    high = average >= 78
    offer = rng.random(n) < np.clip(0.03 + 0.4 * _sigmoid((average - 84) / 3) + 0.05 * olympiads, 0, 0.8)
    visited = rng.poisson(0.3 + 1.2 * offer + 0.6 * (~offers_path & high))
    satisfaction = _scale(rng, 3.5 - 0.7 * (~offers_path & high) - 0.4 * bullying + 0.25 * (teacher_rating - 3.6), n)
    moving = rng.random(n) < 0.04
    phone = np.round(rng.gamma(2.2, 1.4, n), 1)
    struggle = (-3.15 - 8.5 * (attendance - 0.88) - 0.035 * (average - 64) + 0.45 * failed + 0.25 * incidents
                + 0.45 * (trend == 'Declining') - 0.25 * (trend == 'Improving') - 0.35 * extra - 0.2 * sports
                + 0.02 * distance - 0.5 * scholarship + 0.65 * (fee == 'Late') + 1.4 * (fee == 'Unpaid')
                + 0.45 * (parents == 'Low') - 0.3 * (parents == 'High') - 0.08 * counselling + 0.4 * health
                + 0.6 * bullying - 0.2 * (peers - 3) - 0.15 * (teacher_rating - 3) + 0.3 * (income == 'Low')
                + 0.25 * first_gen - 0.5 * boarder + 0.05 * (phone - 3) - 0.6 * (homework - 0.8)
                + rng.normal(0, 0.35, n))
    transfer = (-4.0 + 0.16 * np.maximum(average - 74, 0) - 2.0 * (average < 70)
                + 1.7 * (~offers_path) * (average >= 74) + 1.6 * offer + 0.5 * (coaching & ~offers_path)
                - 0.4 * (satisfaction - 3) + 0.35 * (income == 'High') + 0.35 * np.minimum(visited, 3)
                - 0.3 * career_counselling + 1.2 * moving + rng.normal(0, 0.5, n))
    p = 1 - (1 - _sigmoid(struggle)) * (1 - _sigmoid(transfer))
    left = rng.random(n) < p
    left = np.where(rng.random(n) < 0.03, ~left, left)
    return pd.DataFrame({
        'StudentID': [f'S{i:05d}' for i in range(1, n + 1)],
        'Gender': _pick(rng, ['Female', 'Male'], n),       # no built-in effect (for the fairness check)
        'Age': age, 'Grade': grade, 'Stream': stream, 'Board': _pick(rng, ['CBSE', 'ICSE', 'State'], n, [0.55, 0.2, 0.25]),
        'Section': _pick(rng, list('ABCDE'), n), 'HouseholdIncomeBand': income, 'ParentEducation': parent_ed,
        'FirstGeneration': _yes_no(first_gen), 'Siblings': rng.choice([0, 1, 2, 3, 4], n, p=[0.2, 0.42, 0.25, 0.1, 0.03]),
        'Boarder': _yes_no(boarder), 'DistanceFromSchoolKm': distance, 'TransportMode': transport,
        'AttendanceRate': attendance, 'LateArrivalsPerTerm': late, 'AverageGrade': average, 'MathsScore': maths,
        'ScienceScore': science, 'EnglishScore': english, 'SocialStudiesScore': social, 'SecondLanguageScore': language,
        'ClassPercentile': percentile, 'GradeTrend': trend, 'FailedSubjects': failed,
        'HomeworkCompletionRate': homework, 'StudyHoursPerWeek': study, 'LibraryVisitsPerMonth': library,
        'OnlineLearningHoursPerWeek': online, 'DisciplinaryIncidents': incidents, 'Extracurricular': _yes_no(extra),
        'SportsTeam': _yes_no(sports), 'CompetitionsEntered': olympiads, 'Scholarship': _yes_no(scholarship),
        'FeePaymentStatus': fee, 'ParentalInvolvement': parents, 'ParentTeacherMeetingsAttended': ptm,
        'CounsellingSessions': counselling, 'HealthIssues': _yes_no(health), 'SickDaysLastTerm': sick_days,
        'PeerRelationships': peers, 'BullyingReported': _yes_no(bullying), 'TeacherSupportRating': teacher_rating,
        'SchoolSatisfaction': satisfaction, 'CareerGoal': goal, 'SchoolOffersCareerPath': _yes_no(offers_path),
        'CompetitiveExamCoaching': _yes_no(coaching), 'CareerCounsellingAtSchool': _yes_no(career_counselling),
        'OfferFromAnotherSchool': _yes_no(offer), 'OtherSchoolsVisited': visited,
        'FamilyRelocating': _yes_no(moving), 'PhoneHoursPerDay': phone,
        'DroppedOut': _yes_no(left),
    })


def make_teachers(n=10500, seed=43):
    rng = np.random.default_rng(seed)
    age = rng.integers(23, 63, n)
    experience = np.clip(age - 23 - rng.integers(0, 8, n), 0, None)
    at_school = np.clip(np.round(experience * rng.beta(2, 2.5, n)), 0, None).astype(int)
    qualification = _pick(rng, ['Bachelor', 'Master', 'PhD'], n, [0.45, 0.48, 0.07])
    level = _pick(rng, ['Primary', 'Middle', 'Secondary'], n, [0.32, 0.3, 0.38])
    subject = _pick(rng, ['Maths', 'Science', 'English', 'History', 'Languages', 'Arts', 'PE', 'Computing'], n)
    contract = _pick(rng, ['Permanent', 'Temporary'], n, [0.75, 0.25])
    salary_band = np.clip(1 + experience // 8 + rng.integers(0, 2, n) + (qualification == 'PhD'), 1, 5)
    salary = np.round(28000 + 9000 * salary_band + rng.normal(0, 3500, n), -2)
    since_raise = rng.integers(0, 7, n)
    promotions = rng.poisson(np.clip(0.3 + 0.03 * at_school, 0, None))
    hod = (rng.random(n) < np.clip(0.02 + 0.01 * experience, 0, 0.3))
    periods = np.clip(np.round(rng.normal(28, 4, n) - 6 * hod), 12, 40).astype(int)
    workload = np.round(rng.normal(47, 6, n) + 0.4 * (periods - 28), 1)
    extra_duties = rng.random(n) < 0.4
    overtime = np.clip(np.round(rng.gamma(2, 2.2, n) + 3 * extra_duties), 0, 30).astype(int)
    class_size = rng.integers(16, 42, n)
    admin = rng.integers(1, 6, n)
    principal = _scale(rng, 3.4 + 0.3 * (admin - 3), n)
    colleagues = _scale(rng, 3.7, n)
    behaviour = rng.integers(1, 6, n)
    complaints = rng.poisson(np.clip(0.8 - 0.15 * (behaviour - 3), 0.1, None))
    pd_hours = rng.integers(0, 61, n)
    mentor = rng.random(n) < np.where(experience < 3, 0.6, 0.15)
    performance = _scale(rng, 3.5 + 0.2 * (pd_hours > 30), n)
    results_trend = _pick(rng, ['Improving', 'Stable', 'Declining'], n, [0.3, 0.5, 0.2])
    awards = rng.poisson(0.2 + 0.1 * (performance >= 4))
    commute = np.round(rng.gamma(2, 7, n), 1)
    housing = rng.random(n) < 0.15
    hometown = np.round(rng.gamma(1.4, 120, n))
    leave_days = rng.poisson(9, n)
    burnout = np.clip(np.round(4 + 0.15 * (workload - 47) + 0.12 * overtime + 0.06 * (class_size - 28)
                               - 0.4 * (admin - 3) + rng.normal(0, 1.4, n)), 1, 10).astype(int)
    sick_days = rng.poisson(np.clip(3 + 0.5 * (burnout - 4), 0.5, None))
    balance = _scale(rng, 3.3 - 0.3 * (burnout - 4), n)
    satisfaction = np.clip(np.round(3 + 0.35 * (admin - 3) - 0.06 * (workload - 47) + 0.25 * (behaviour - 3)
                                    - 0.2 * (burnout - 4) + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    resources = _scale(rng, 3.3, n)
    tech = _scale(rng, 3.4, n)
    growth = _scale(rng, 3.2 + 0.4 * (promotions > 0) + 0.2 * (pd_hours > 30), n)
    offer = rng.random(n) < np.clip(0.05 + 0.04 * (performance - 3) + 0.03 * (qualification != 'Bachelor'), 0.01, 0.4)
    searching = rng.random(n) < np.clip(0.06 + 0.05 * (burnout - 4) + 0.08 * (satisfaction <= 2), 0.01, 0.7)
    budget = np.clip(np.round(rng.beta(2, 2, n) * 100), 0, 100).astype(int)
    own_children = rng.random(n) < 0.25
    spouse = rng.random(n) < 0.05
    observation = _scale(rng, 3.4 + 0.3 * (performance - 3), n, 1, 10, 1.5)
    meetings = rng.poisson(3, n)
    clubs = rng.poisson(0.8, n)
    exam_duties = rng.poisson(6, n)
    logit = (-2.75 + 0.08 * (workload - 47) + 0.05 * (class_size - 28) + 0.25 * since_raise - 0.35 * (admin - 3)
             - 0.025 * pd_hours + 0.02 * commute + 1.1 * (contract == 'Temporary') - 0.45 * (satisfaction - 3)
             - 0.25 * (behaviour - 3) - 0.25 * (salary_band - 3) + 0.8 * (experience < 3) + 0.7 * (experience > 33)
             + 0.18 * (burnout - 4) + 0.04 * overtime - 0.25 * (principal - 3) - 0.2 * (colleagues - 3)
             - 0.35 * mentor * (experience < 3) + 1.3 * offer + 1.1 * searching + 1.4 * spouse - 0.5 * housing
             - 0.4 * own_children - 0.25 * (growth - 3) + 0.0015 * hometown - 0.25 * promotions
             + rng.normal(0, 0.5, n))
    left = _outcome(rng, logit)
    return pd.DataFrame({
        'TeacherID': [f'T{i:05d}' for i in range(1, n + 1)],
        'Gender': _pick(rng, ['Female', 'Male'], n, [0.6, 0.4]),   # no built-in effect
        'Age': age, 'MaritalStatus': _pick(rng, ['Single', 'Married', 'Divorced'], n, [0.3, 0.6, 0.1]),
        'YearsExperience': experience, 'YearsAtThisSchool': at_school, 'Qualification': qualification,
        'CertifiedTeacher': _yes_no(rng.random(n) < 0.85), 'Subject': subject, 'GradeLevelTaught': level,
        'HeadOfDepartment': _yes_no(hod), 'ContractType': contract, 'SalaryBand': [f'Band {b}' for b in salary_band],
        'AnnualSalary': salary, 'YearsSinceLastRaise': since_raise, 'PromotionsLast5Years': promotions,
        'TeachingPeriodsPerWeek': periods, 'WorkloadHoursPerWeek': workload, 'ExtraDuties': _yes_no(extra_duties),
        'OvertimeHoursPerWeek': overtime, 'ClassSize': class_size, 'AdminSupportRating': admin,
        'PrincipalRelationship': principal, 'ColleagueSupport': colleagues, 'StudentBehaviourRating': behaviour,
        'ParentComplaintsLastYear': complaints, 'ProfessionalDevelopmentHours': pd_hours, 'MentorAssigned': _yes_no(mentor),
        'PerformanceRating': performance, 'ClassObservationScore': observation, 'StudentResultsTrend': results_trend,
        'TeachingAwards': awards, 'CommuteKm': commute, 'HousingProvided': _yes_no(housing),
        'HometownDistanceKm': hometown, 'LeaveDaysTaken': leave_days, 'SickDaysLastYear': sick_days,
        'BurnoutScore': burnout, 'WorkLifeBalance': balance, 'JobSatisfaction': satisfaction,
        'ResourcesAdequacy': resources, 'TechnologySupport': tech, 'CareerGrowthRating': growth,
        'OfferFromAnotherSchool': _yes_no(offer), 'ActivelyJobSearching': _yes_no(searching),
        'TrainingBudgetUsedPct': budget, 'OwnChildrenAtSchool': _yes_no(own_children),
        'SpouseRelocating': _yes_no(spouse), 'MeetingsPerWeek': meetings, 'AfterSchoolClubsRun': clubs,
        'ExamDutiesPerYear': exam_duties, 'StudentBehaviourIncidents': rng.poisson(np.clip(4 - 0.7 * (behaviour - 3), 0.5, None)),
        'Left': left,
    })


def make_applicants(n=11000, seed=44):
    rng = np.random.default_rng(seed)
    grade = rng.integers(6, 12, n)
    age = grade + 5 + rng.integers(0, 2, n)
    income = _pick(rng, ['Low', 'Middle', 'High'], n, [0.25, 0.5, 0.25])
    previous = _pick(rng, ['Government', 'Private', 'International'], n, [0.35, 0.55, 0.10])
    prev_result = np.clip(np.round(rng.normal(72, 12, n)), 30, 100)
    score = np.clip(np.round(prev_result * 0.6 + rng.normal(28, 9, n)), 20, 100)
    interview = np.clip(np.round(rng.normal(6.5, 1.6, n) + 0.03 * (score - 70), 1), 1, 10)
    maths = np.clip(np.round(score + rng.normal(0, 8, n)), 10, 100)
    english = np.clip(np.round(score + rng.normal(0, 7, n)), 10, 100)
    stream = np.where(grade >= 10, _pick(rng, ['Science', 'Commerce', 'Humanities'], n, [0.5, 0.3, 0.2]), 'General')
    goal = _pick(rng, ['Engineering', 'Medicine', 'Computer Science', 'Business', 'Law', 'Arts & Design', 'Undecided'], n)
    offers_stream = rng.random(n) < np.where(stream == 'Science', 0.75, 0.9)
    distance = np.round(rng.gamma(2, 12, n), 1)
    hostel_needed = distance > 45
    hostel_offered = hostel_needed & (rng.random(n) < 0.6)
    transport = (distance <= 45) & (rng.random(n) < 0.75)
    scholarship = rng.random(n) < np.clip(0.08 + 0.012 * (score - 68), 0.02, 0.6)
    scholarship_pct = np.where(scholarship, rng.choice([10, 25, 50, 75, 100], n, p=[0.3, 0.3, 0.25, 0.1, 0.05]), 0)
    fee = _pick(rng, ['Low', 'Medium', 'High'], n, [0.3, 0.45, 0.25])
    plan = rng.random(n) < 0.35
    visit = rng.random(n) < 0.35
    open_day = rng.random(n) < 0.25
    tour = rng.random(n) < 0.3
    siblings = rng.choice([0, 1, 2], n, p=[0.75, 0.2, 0.05])
    alumni = rng.random(n) < 0.1
    followups = rng.integers(0, 7, n)
    emails = rng.poisson(3 + 0.5 * followups)
    web = rng.poisson(4, n)
    brochure = rng.random(n) < 0.4
    source = _pick(rng, ['Website', 'School fair', 'Referral', 'Agent', 'Social media'], n, [0.35, 0.2, 0.15, 0.1, 0.2])
    month = _pick(rng, ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], n)
    days = rng.integers(1, 180, n)
    other_apps = rng.poisson(1.8 + 0.02 * (score - 68))
    other_offer = rng.random(n) < np.clip(0.1 + 0.1 * other_apps + 0.01 * (score - 68), 0.02, 0.9)
    rival_rank = np.where(other_offer, rng.integers(1, 6, n), 0)
    reviews = rng.poisson(2, n)
    counsellor = rng.random(n) < 0.45
    docs = rng.random(n) < 0.85
    fee_paid = rng.random(n) < 0.7
    early = rng.random(n) < 0.2
    waitlisted = rng.random(n) < 0.12
    logit = (-1.85 + 1.2 * scholarship + 0.012 * scholarship_pct + 1.4 * visit + 0.5 * open_day + 0.3 * tour
             - 0.03 * distance + 0.8 * (siblings > 0) + 0.7 * alumni + 0.28 * np.minimum(followups, 4)
             - 0.75 * (fee == 'High') + 0.35 * (fee == 'Low') + 0.4 * plan + 0.55 * (source == 'Referral')
             + 0.25 * (source == 'School fair') - 0.3 * (source == 'Social media') - 0.02 * (score - 68)
             - 1.2 * other_offer + 0.2 * other_offer * (rival_rank <= 2) - 0.9 * (~offers_stream)
             + 0.6 * hostel_offered - 0.8 * (hostel_needed & ~hostel_offered) + 0.3 * transport
             + 0.5 * counsellor + 0.35 * early - 0.4 * waitlisted + 0.9 * docs + 0.7 * fee_paid - 0.004 * days + 0.05 * np.minimum(emails, 8)
             + rng.normal(0, 0.5, n))
    return pd.DataFrame({
        'ApplicantID': [f'A{i:05d}' for i in range(1, n + 1)],
        'Gender': _pick(rng, ['Female', 'Male'], n),       # no built-in effect
        'Age': age, 'ApplyingForGrade': grade, 'PreferredStream': stream, 'CareerGoal': goal,
        'SchoolOffersPreferredStream': _yes_no(offers_stream), 'PreviousSchoolType': previous,
        'PreviousBoard': _pick(rng, ['CBSE', 'ICSE', 'State', 'IB'], n, [0.5, 0.2, 0.25, 0.05]),
        'PreviousResultPct': prev_result, 'EntranceScore': score, 'InterviewScore': interview,
        'MathsTestScore': maths, 'EnglishTestScore': english, 'HouseholdIncomeBand': income,
        'ParentOccupation': _pick(rng, ['Salaried', 'Business', 'Government', 'Self-employed', 'Farming'], n),
        'AlumniParent': _yes_no(alumni), 'SiblingsEnrolled': siblings, 'DistanceKm': distance,
        'TransportAvailable': _yes_no(transport), 'HostelNeeded': _yes_no(hostel_needed),
        'HostelOffered': _yes_no(hostel_offered), 'FeeLevel': fee, 'ScholarshipOffered': _yes_no(scholarship),
        'ScholarshipPercent': scholarship_pct, 'PaymentPlanOffered': _yes_no(plan), 'CampusVisit': _yes_no(visit),
        'OpenDayAttended': _yes_no(open_day), 'VirtualTourViewed': _yes_no(tour), 'FollowUpContacts': followups,
        'EmailsOpened': emails, 'WebsiteVisits': web, 'BrochureDownloaded': _yes_no(brochure),
        'ApplicationSource': source, 'ApplicationMonth': month, 'DaysSinceApplication': days,
        'OtherSchoolsAppliedTo': other_apps, 'OfferFromOtherSchool': _yes_no(other_offer),
        'RivalSchoolRanking': rival_rank, 'ParentReviewsRead': reviews, 'CounsellorCallDone': _yes_no(counsellor),
        'DocumentsComplete': _yes_no(docs), 'ApplicationFeePaid': _yes_no(fee_paid),
        'SportsQuota': _yes_no(rng.random(n) < 0.06), 'AdmissionTestAttempt': rng.choice([1, 2], n, p=[0.85, 0.15]),
        'SiblingApplyingToo': _yes_no(rng.random(n) < 0.05), 'PreferredLanguage': _pick(rng, ['English', 'Hindi', 'Other'], n, [0.7, 0.2, 0.1]),
        'NewsletterSubscribed': _yes_no(rng.random(n) < 0.3), 'EarlyBirdDiscount': _yes_no(early),
        'InterviewInPerson': _yes_no(rng.random(n) < 0.6), 'WaitlistedFirst': _yes_no(waitlisted),
        'Enrolled': _outcome(rng, logit),
    })


def _enlarge(df, target, id_column, n_total, rng, make_id):
    """Keep every real row, then add made-up rows. Each made-up row starts from a random real row with small changes
    to its numbers; its outcome is NOT copied but drawn from the chance a simple model (trained on the real rows) gives
    it. Copied outcomes would let a model memorise near-duplicate rows and score unrealistically well."""
    extra = df.sample(n_total - len(df), replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    for col in extra.columns:
        if col in (target, id_column) or not pd.api.types.is_numeric_dtype(extra[col]):
            continue
        values = extra[col].astype(float)
        low, high = df[col].min(), df[col].max()
        if df[col].nunique() <= 10:          # ratings and levels: sometimes move one step
            step = rng.choice([-1, 0, 0, 0, 1], len(extra))
            extra[col] = np.clip(values + step, low, high).astype(df[col].dtype)
        else:                                 # amounts: up to about 10% up or down
            jittered = np.clip(values * rng.normal(1, 0.1, len(extra)), low, high)
            extra[col] = jittered.round(0 if pd.api.types.is_integer_dtype(df[col]) else 2).astype(df[col].dtype)
    features = [c for c in df.columns if c not in (target, id_column)]
    both = pd.get_dummies(pd.concat([df[features], extra[features]], ignore_index=True).apply(
        lambda c: pd.to_numeric(c, errors='coerce').fillna(0) if c.name in df.select_dtypes('number').columns
        or pd.to_numeric(c, errors='coerce').notna().mean() > 0.95 else c), dtype=float)
    both = (both - both.mean()) / both.std().replace(0, 1)
    model = LogisticRegression(max_iter=2000, C=0.5).fit(both.iloc[:len(df)], df[target] == 'Yes')
    p = model.predict_proba(both.iloc[len(df):])[:, 1]
    rate = float((df[target] == 'Yes').mean())
    p = rate + 0.6 * (p - rate)              # softer than the model, like real data (no perfect predictions)
    extra[target] = _yes_no(rng.random(len(extra)) < p)
    extra[id_column] = [make_id(i) for i in range(len(extra))]
    return pd.concat([df, extra], ignore_index=True)


def make_employees_large(n=10500, seed=45):
    """The IBM HR attrition data (every real row) plus made-up columns and made-up extra rows."""
    rng = np.random.default_rng(seed)
    base = tidy_employees(pd.read_csv(find_sample('employees.csv')))
    start = int(base['EmployeeNumber'].max()) + 1
    df = _enlarge(base, 'Attrition', 'EmployeeNumber', n, rng, lambda i: start + i)
    m = len(df)
    left = (df['Attrition'] == 'Yes').to_numpy()
    overtime = (df['OverTime'] == 'Yes').to_numpy()
    sat = df['JobSatisfaction'].to_numpy()
    env = df['EnvironmentSatisfaction'].to_numpy()
    wlb = df['WorkLifeBalance'].to_numpy()
    income = df['MonthlyIncome'].to_numpy()
    years = df['YearsAtCompany'].to_numpy()
    df['RemoteDaysPerWeek'] = np.clip(rng.poisson(1.6 - 0.5 * left), 0, 5)
    df['CommuteMinutes'] = np.clip(np.round(df['DistanceFromHome'] * 2.4 + rng.normal(8, 6, m)), 5, 150).astype(int)
    df['EngagementScore'] = np.clip(np.round(6.5 + 0.4 * (sat - 2.5) + 0.3 * (env - 2.5) - 0.6 * left + rng.normal(0, 1.4, m)), 1, 10).astype(int)
    df['ManagerRating'] = _scale(rng, 3.5 + 0.25 * (df['RelationshipSatisfaction'].to_numpy() - 2.5) - 0.2 * left, m)
    df['PeerFeedbackScore'] = _scale(rng, 3.7 - 0.1 * left, m)
    df['TrainingHoursLastYear'] = np.clip(np.round(df['TrainingTimesLastYear'] * 6 + rng.normal(4, 5, m)), 0, 120).astype(int)
    df['ProjectsLastYear'] = rng.poisson(3 + 0.5 * (df['JobLevel'].to_numpy() > 2))
    df['OvertimeHoursPerMonth'] = np.clip(np.round(rng.gamma(2, 3, m) + 18 * overtime + 2 * left), 0, 90).astype(int)
    df['SickDaysLastYear'] = rng.poisson(4 + 0.8 * left + 1.2 * (wlb <= 2))
    df['LeaveDaysTaken'] = np.clip(rng.poisson(14 - 3 * overtime), 0, 30)
    df['RecruiterContactsLastYear'] = rng.poisson(1 + 0.00012 * income + 0.9 * left)
    df['InternalJobApplications'] = rng.poisson(0.4 + 0.15 * df['YearsSinceLastPromotion'].to_numpy() * 0.3 + 0.15 * left)
    df['BonusLastYearPct'] = np.clip(np.round(rng.normal(8, 4, m) + 1.5 * (df['PerformanceRating'].to_numpy() - 3) - 1 * left, 1), 0, 30)
    df['SalaryVsMarketPct'] = np.clip(np.round(rng.normal(100, 9, m) + 0.0006 * (income - 6500) - 3 * left), 60, 140).astype(int)
    df['MentorAssigned'] = _yes_no(rng.random(m) < np.where(years < 2, 0.5, 0.15) - 0.05 * left)
    df['CareerPathDiscussed'] = _yes_no(rng.random(m) < 0.55 - 0.12 * left)
    df['TeamSize'] = np.clip(rng.poisson(9, m), 2, 40)
    df['ManagerChangesLast2Years'] = rng.poisson(0.7 + 0.3 * left)
    df['FlexibleHours'] = _yes_no(rng.random(m) < 0.45 - 0.08 * left)
    df['CertificationsEarned'] = rng.poisson(1.0 + 0.2 * df['Education'].to_numpy() * 0.5)
    df['HealthBenefitsUsed'] = _yes_no(rng.random(m) < 0.6)
    target = df.pop('Attrition')
    df['Attrition'] = target
    return df


def make_customers_large(n=10500, seed=46):
    """The IBM Telco churn data (every real row) plus made-up columns and made-up extra rows."""
    rng = np.random.default_rng(seed)
    base = pd.read_csv(find_sample('customers.csv'))
    letters = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    def new_id(i):
        return f"{rng.integers(1000, 9999)}-{''.join(rng.choice(letters, 5))}"
    df = _enlarge(base, 'Churn', 'customerID', n, rng, new_id)
    m = len(df)
    churn = (df['Churn'] == 'Yes').to_numpy()
    tenure = df['tenure'].to_numpy()
    monthly = df['MonthlyCharges'].to_numpy()
    month_to_month = (df['Contract'] == 'Month-to-month').to_numpy()
    fiber = (df['InternetService'] == 'Fiber optic').to_numpy()
    no_internet = (df['InternetService'] == 'No').to_numpy()
    senior = df['SeniorCitizen'].to_numpy() == 1
    df['Age'] = np.where(senior, rng.integers(65, 86, m), rng.integers(19, 65, m))
    df['Region'] = _pick(rng, ['North', 'South', 'East', 'West'], m)
    df['CityTier'] = _pick(rng, ['Tier 1', 'Tier 2', 'Tier 3'], m, [0.45, 0.35, 0.2])
    df['HouseholdSize'] = np.clip(rng.poisson(2.6, m), 1, 8)
    df['AvgMonthlyDataGB'] = np.where(no_internet, 0, np.round(rng.gamma(2.2, 18, m) + 25 * fiber, 1))
    df['AvgCallMinutes'] = np.round(rng.gamma(2, 120, m))
    df['InternationalCalls'] = _yes_no(rng.random(m) < 0.15)
    df['NetworkOutagesLastQuarter'] = rng.poisson(0.8 + 0.5 * churn + 0.6 * fiber)
    df['AvgDownloadSpeedMbps'] = np.where(no_internet, 0, np.round(np.clip(rng.normal(60 + 120 * fiber, 25, m) - 6 * churn, 5, None)))
    df['SupportTicketsLastYear'] = rng.poisson(1 + 0.6 * churn + 0.4 * fiber)
    df['AvgResolutionDays'] = np.round(rng.gamma(2, 1.2, m) + 0.5 * churn, 1)
    df['ComplaintsLastYear'] = rng.poisson(0.3 + 0.4 * churn)
    df['SatisfactionScore'] = np.clip(np.round(7.2 - 0.9 * churn + 0.02 * tenure / 6 + rng.normal(0, 1.6, m)), 1, 10).astype(int)
    df['NPSScore'] = np.clip(np.round(df['SatisfactionScore'] + rng.normal(0, 1.2, m)), 0, 10).astype(int)
    df['LatePaymentsLastYear'] = rng.poisson(0.5 + 0.4 * churn)
    df['DiscountReceived'] = _yes_no(rng.random(m) < np.where(month_to_month, 0.15, 0.4))
    df['LoyaltyPoints'] = np.round(tenure * rng.uniform(20, 40, m) * (1 - 0.05 * churn)).astype(int)
    df['AppLoginsPerMonth'] = rng.poisson(np.clip(6 - 1.2 * churn + 0.03 * tenure, 0.3, None))
    df['MonthsSinceLastPlanChange'] = np.clip(np.round(rng.gamma(2, 6, m)), 0, tenure.clip(0, None) + 1).astype(int)
    df['PriceIncreaseLastYear'] = _yes_no(rng.random(m) < 0.3 + 0.12 * churn)
    df['CompetitorOfferReceived'] = _yes_no(rng.random(m) < 0.15 + 0.18 * churn * month_to_month + 0.05 * churn)
    df['ReferralsMade'] = rng.poisson(0.5 + 0.2 * (~churn))
    df['DevicesConnected'] = np.where(no_internet, 0, np.clip(rng.poisson(4, m), 1, 15))
    df['StreamingHoursPerWeek'] = np.where(df['StreamingTV'] == 'Yes', np.round(rng.gamma(2, 5, m), 1), np.round(rng.gamma(1, 1.5, m), 1))
    df['PromotionsUsed'] = rng.poisson(1.2 - 0.2 * churn + 0.4 * (~month_to_month)).clip(0)
    df['BillShockEvents'] = rng.poisson(0.2 + 0.3 * churn + 0.004 * (monthly - 65).clip(0))
    df['HasMobileBundle'] = _yes_no(rng.random(m) < 0.35 - 0.07 * churn)
    df['ContractRenewalDue'] = _yes_no(rng.random(m) < np.where(month_to_month, 0.0, 0.12))
    df['DaysSinceLastContact'] = rng.integers(0, 365, m)
    df['EmailOptIn'] = _yes_no(rng.random(m) < 0.55)
    df['CustomerServiceCallsLastMonth'] = rng.poisson(0.3 + 0.3 * churn)
    target = df.pop('Churn')
    df['Churn'] = target
    return df


SYNTHETIC_MAKERS = {
    'students_SYNTHETIC.csv': make_students,
    'teachers_SYNTHETIC.csv': make_teachers,
    'custom_join_SYNTHETIC.csv': make_applicants,
    'employees_SYNTHETIC.csv': make_employees_large,
    'customers_SYNTHETIC.csv': make_customers_large,
}


def tidy_employees(df):
    """IBM HR file: drop the columns that have the same value in every row."""
    df = df.rename(columns=lambda c: str(c).replace('﻿', '').strip())
    return df.drop(columns=[c for c in ['EmployeeCount', 'Over18', 'StandardHours'] if c in df.columns])


def find_sample(file_name, ask_upload=True):
    """Path of a sample file: found, generated, downloaded or uploaded (in that order)."""
    for folder in SAMPLE_FOLDERS:
        path = os.path.join(folder, file_name)
        if os.path.exists(path):
            return path
    for folder in SAMPLE_FOLDERS:
        for pattern in SAMPLE_ALIASES.get(file_name, []):
            matches = sorted(glob.glob(os.path.join(folder, pattern)))
            if matches:
                return matches[0]
    os.makedirs('sample_data', exist_ok=True)
    target = os.path.join('sample_data', file_name)
    if file_name in SYNTHETIC_MAKERS:
        SYNTHETIC_MAKERS[file_name]().to_csv(target, index=False)
        return target
    if file_name in SAMPLE_URLS:
        try:
            df = pd.read_csv(SAMPLE_URLS[file_name])
            (tidy_employees(df) if file_name == 'employees.csv' else df).to_csv(target, index=False)
            return target
        except Exception as error:
            print(f'Could not download {file_name}: {error}')
    if IN_COLAB and ask_upload:
        print(f'Please upload {file_name}:')
        uploaded = files.upload()
        names = [n for n in uploaded if n.lower().endswith(('.csv', '.xlsx'))]
        if names:
            return names[0]
    return None





LEVEL_COLOURS = {
    'leave': {'High': '#d62728', 'Medium': '#ff9f1c', 'Low': '#2ca02c'},   # high risk = bad = red
    'join': {'High': '#2ca02c', 'Medium': '#ff9f1c', 'Low': '#9e9e9e'},    # high likelihood = good = green
}
BEST_COLOUR = '#ffb000'   # star marker for best values
BASE_COLOUR = '#4c78a8'


def make_wording(scenario, mode=None, **overrides):
    """Wording for a scenario. mode switches a Custom scenario between 'leave' and 'join'; overrides replace words."""
    sc = SCENARIOS[scenario]
    w = {k: sc[k] for k in ['mode', 'person', 'people', 'org', 'outcome_past', 'outcome_ing', 'stay_past']}
    if mode and mode != w['mode']:
        w.update(CUSTOM_WORDING[mode])
        w['mode'] = mode
    w.update({k: str(v).strip() for k, v in overrides.items() if v is not None and str(v).strip()})
    return w


def words(w):
    """Phrases derived from the wording."""
    leave = w['mode'] == 'leave'
    score = f"risk of {w['outcome_ing']}" if leave else f"likelihood of {w['outcome_ing']}"
    return {
        'leave': leave,
        'score': score,
        'Score': score[0].upper() + score[1:],
        'why': 'Why They Leave' if leave else 'Why They Join',
        'list': 'at-risk list' if leave else 'most-likely-to-join list',
        'list_file': 'at_risk_list' if leave else 'most_likely_to_join_list',
        'actions': 'retention' if leave else 'conversion and outreach',
        'Actions': 'Retention' if leave else 'Conversion and outreach',
        'flagged': 'flagged as at risk' if leave else 'predicted to join',
        'top_group': 'highest-risk group' if leave else 'most-likely group',
        'pushes': f"towards {w['outcome_ing']}",
        'colours': LEVEL_COLOURS[w['mode']],
        'highlight': LEVEL_COLOURS[w['mode']]['High'],
        'outcome_rate': f"% who {w['outcome_past']}",
    }


def pct(x, digits=0):
    return f'{100 * x:.{digits}f}%'


def one_in(share):
    """'about 1 in 3' for a share such as 0.33."""
    if share <= 0:
        return 'none'
    return f'about 1 in {max(1, round(1 / share))}'


def safe_name(text):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')


def load_table(path):
    """Read a CSV or Excel (.xlsx) file into a DataFrame."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.xlsx':
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload a .csv or .xlsx file.")
    df.columns = [str(c).replace('﻿', '').strip() for c in df.columns]
    if df.shape[0] == 0 or df.shape[1] < 2:
        raise ValueError('The file needs at least one data row and at least two columns.')
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f'The file has duplicated column names: {duplicated}')
    return df


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def convert_numeric_text(df, min_share=0.9):
    """Convert text columns whose non-blank values are (almost) all numbers, e.g. '29.85'. Blank text becomes missing."""
    df = df.copy()
    converted = []
    for col in df.columns:
        s = df[col]
        if is_numeric(s) or pd.api.types.is_bool_dtype(s):
            continue
        text = s.astype(str).str.strip()
        blank = s.isna() | text.eq('')
        if (~blank).sum() == 0:
            continue
        numbers = pd.to_numeric(text.where(~blank), errors='coerce').astype(float)
        if numbers[~blank].notna().mean() >= min_share:
            df[col] = numbers
            converted.append(col)
    return df, converted


def numeric_columns(df):
    converted, _ = convert_numeric_text(df)
    return [c for c in converted.columns if is_numeric(converted[c])]


def column_words(name):
    """'SeniorCitizen' -> ['senior', 'citizen']; 'marital_status' -> ['marital', 'status']."""
    return [w.lower() for w in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+', str(name))]


def find_sensitive_columns(df, scenario=None):
    listed = SCENARIOS[scenario]['sensitive'] if scenario in SCENARIOS else []
    return [c for c in df.columns if c in listed or SENSITIVE_WORDS & set(column_words(c))]


def find_personal_columns(df):
    """Columns that look like direct personal identifiers: named like a name or contact detail AND mostly unique
    (so 'PhoneService' with values Yes/No is not flagged, but 'Phone' with a different number per row is)."""
    personal = {'name', 'surname', 'firstname', 'lastname', 'email', 'mail', 'phone', 'mobile', 'telephone',
                'address', 'postcode', 'zip'}
    found = []
    for c in df.columns:
        values = df[c].dropna()
        if personal & set(column_words(c)) and len(values) and values.nunique() > 0.5 * len(values):
            found.append(c)
    return found


def guess_id_columns(df, scenario=None):
    """Columns to suggest dropping: listed IDs, columns named like an ID, and text that is (almost) unique per row."""
    listed = SCENARIOS[scenario]['id_columns'] if scenario in SCENARIOS else []
    converted, _ = convert_numeric_text(df)
    guesses = []
    for col in df.columns:
        name = col.lower().replace('_', '').replace(' ', '')
        values = converted[col].dropna()
        mostly_unique_text = (not is_numeric(converted[col]) and len(values) > 20 and values.nunique() > 50
                              and values.nunique() > 0.5 * len(values))
        if col in listed or name == 'id' or name.endswith('id') or mostly_unique_text:
            guesses.append(col)
    return guesses + [c for c in find_personal_columns(df) if c not in guesses]


def guess_target(df, scenario=None):
    candidates = SCENARIOS[scenario]['targets'] if scenario in SCENARIOS else []
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    two_valued = [c for c in df.columns if df[c].nunique() == 2]
    return two_valued[-1] if two_valued else df.columns[-1]


def value_to_text(value):
    """Turn a target value into text, so 1, 1.0 and '1' are the same class."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def target_classes(df, target):
    return sorted(df[target].map(value_to_text).dropna().unique().tolist())


def default_positive_class(df, target):
    """'Yes'/'1'/'True' when present, otherwise the least common class."""
    classes = target_classes(df, target)
    for candidate in ['Yes', 'yes', 'Y', '1', 'True', 'true', 'Dropout', 'Left']:
        if candidate in classes:
            return candidate
    counts = df[target].map(value_to_text).value_counts()
    return counts.index[-1] if len(counts) else None


def describe_columns(df, sensitive=()):
    """One row per column: type, detected kind, missing values, unique values, example."""
    converted, from_text = convert_numeric_text(df)
    rows = []
    for col in df.columns:
        kind = 'numeric (text converted)' if col in from_text else ('numeric' if is_numeric(converted[col]) else 'categorical')
        non_null = df[col].dropna()
        rows.append({
            'Column': col, 'Type': str(df[col].dtype), 'Detected as': kind,
            'Missing': int(converted[col].isna().sum()), 'Missing %': round(100 * converted[col].isna().mean(), 1),
            'Unique values': int(df[col].nunique()), 'Sensitive': 'yes' if col in sensitive else '',
            'Example': str(non_null.iloc[0]) if len(non_null) else '',
        })
    return pd.DataFrame(rows)


def prepare_outcome_data(df, target, features, positive_class, missing_strategy='median',
                         exclude_sensitive=True, sensitive=(), wording=None):
    """Validate and clean the data.

    Returns a dict with X (feature columns used for training), y (1 = outcome happened), `source` (the cleaned rows
    with every column, including excluded sensitive ones, for the fairness check), column groups and notes.
    Raises ValueError with a readable message when the data cannot be used.
    """
    notes = []
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the data.")
    features = [c for c in features if c != target]
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f'These feature columns are not in the data: {missing_cols}')
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f'Missing value strategy must be one of {MISSING_STRATEGIES}.')
    sensitive = [c for c in sensitive if c in df.columns and c != target]
    excluded = [c for c in features if c in sensitive] if exclude_sensitive else []
    if excluded:
        features = [c for c in features if c not in excluded]
        notes.append(f'Sensitive columns left out of training: {excluded}')
    if not features:
        raise ValueError('Choose at least one feature column (other than the target and excluded sensitive columns).')

    keep = list(dict.fromkeys(features + sensitive))
    data, converted = convert_numeric_text(df[keep])
    converted = [c for c in converted if c in features]
    if converted:
        notes.append(f'Converted text columns to numbers: {converted}')
    data['__outcome__'] = df[target].map(value_to_text)

    no_target = int(data['__outcome__'].isna().sum())
    if no_target:
        data = data[data['__outcome__'].notna()]
        notes.append(f'Dropped {no_target} rows with a missing target value.')
    empty_cols = [c for c in features if data[c].isna().all()]
    if empty_cols:
        features = [c for c in features if c not in empty_cols]
        notes.append(f'Ignored columns with no values: {empty_cols}')
        if not features:
            raise ValueError('All chosen feature columns are empty.')
    if missing_strategy == 'drop rows':
        before = len(data)
        data = data.dropna(subset=features)
        if before - len(data):
            notes.append(f'Dropped {before - len(data)} rows with missing values.')
    if len(data) < MIN_ROWS:
        raise ValueError(f'Only {len(data)} usable rows. The app needs at least {MIN_ROWS}.')

    classes = sorted(data['__outcome__'].unique().tolist())
    if len(classes) < 2:
        raise ValueError(f"The target '{target}' has only one value ({classes[0]}). It needs at least two.")
    if len(classes) > 20 and is_numeric(df[target]):
        raise ValueError(f"The target '{target}' looks continuous ({len(classes)} different numbers). "
                         'Pick a Yes/No style target, or use the Regression tab for numeric targets.')
    positive_class = value_to_text(positive_class)
    if positive_class not in classes:
        raise ValueError(f"'{positive_class}' is not a value of '{target}'. Choose one of {classes}.")
    if len(classes) > 2:
        notes.append(f"'{target}' has {len(classes)} values; '{positive_class}' counts as the outcome and all others as not.")

    y = (data['__outcome__'] == positive_class).astype(int)
    for label, count in [('the outcome', int(y.sum())), ('no outcome', int((1 - y).sum()))]:
        if count < 5:
            raise ValueError(f'Only {count} rows have {label}. Each side needs at least 5 rows.')
    minority = min(y.mean(), 1 - y.mean())
    if minority < 0.15:
        notes.append(f'The classes are very imbalanced (only {pct(minority, 1)} in the smaller class). '
                     "Keep class_weight='balanced' switched on, and judge models by ROC-AUC or recall, not accuracy.")

    X = data[features].copy()
    numeric = [c for c in features if is_numeric(X[c])]
    categorical = [c for c in features if c not in numeric]
    for col in categorical:
        X[col] = pd.Series([np.nan if pd.isna(v) else str(v) for v in X[col]], index=X.index, dtype=object)
        if X[col].nunique() > 50:
            notes.append(f"Column '{col}' has {X[col].nunique()} different values. If it is an ID or free text, drop it.")
    source = data.drop(columns='__outcome__')
    return {
        'X': X, 'y': y, 'source': source, 'target': target, 'positive_class': positive_class,
        'features': features, 'numeric': numeric, 'categorical': categorical, 'sensitive': sensitive,
        'excluded': excluded, 'missing_strategy': missing_strategy, 'notes': notes,
    }


def clean_new_rows(prepared, rows):
    """Apply the same cleaning to new rows (for predictions). Returns X and notes."""
    notes = []
    missing_cols = [c for c in prepared['features'] if c not in rows.columns]
    if missing_cols:
        raise ValueError(f'The new data is missing these feature columns: {missing_cols}')
    X = rows[prepared['features']].copy()
    for col in prepared['numeric']:
        values = X[col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        numbers = pd.to_numeric(values, errors='coerce').astype(float)
        bad = numbers.isna() & values.notna()
        if bad.any():
            notes.append(f"'{col}': {int(bad.sum())} value(s) are not numbers and were treated as missing.")
        X[col] = numbers
    for col in prepared['categorical']:
        X[col] = pd.Series([np.nan if pd.isna(v) or str(v).strip() == '' else value_to_text(v) for v in X[col]],
                           index=X.index, dtype=object)
    return X, notes


def typical_row(prepared):
    """One row with the median of numeric columns and the most common value of categorical columns."""
    X = prepared['X']
    return pd.DataFrame([{c: (round(float(X[c].median()), 4) if c in prepared['numeric'] else X[c].mode().iloc[0])
                          for c in prepared['features']}])


GRID_RULES = {
    'Logistic Regression': {'C': ('float', lambda v: v > 0, 'a number greater than 0')},
    'Decision Tree': {
        'max_depth': ('int_or_none', lambda v: v is None or v >= 1, 'a whole number >= 1, or None'),
        'min_samples_leaf': ('int', lambda v: v >= 1, 'a whole number >= 1'),
    },
    'K-Nearest Neighbors': {
        'n_neighbors': ('int', lambda v: v >= 1, 'a whole number >= 1'),
        'weights': ('text', lambda v: v in ('uniform', 'distance'), "'uniform' or 'distance'"),
        'p': ('int', lambda v: v in (1, 2), '1 (Manhattan) or 2 (Euclidean)'),
    },
    'Random Forest': {
        'n_estimators': ('int', lambda v: v >= 1, 'a whole number >= 1'),
        'max_depth': ('int_or_none', lambda v: v is None or v >= 1, 'a whole number >= 1, or None'),
        'min_samples_leaf': ('int', lambda v: v >= 1, 'a whole number >= 1'),
    },
    'Neural Network': {
        'hidden_layer_sizes': ('layers', lambda v: all(n >= 1 for n in v), "layer sizes such as 16 or 32x16"),
    },
}
# directions in which a search range can usefully be widened (p only allows 1 or 2)
EXTENDABLE = {'C': 'both', 'max_depth': 'both', 'min_samples_leaf': 'both', 'n_neighbors': 'both',
              'n_estimators': 'up', 'hidden_layer_sizes': 'both'}
PARAM_LABELS = {'C': 'C (regularisation strength, inverse)', 'max_depth': 'max depth', 'min_samples_leaf': 'min samples leaf',
                'n_neighbors': 'K (number of neighbours)', 'weights': 'weight type', 'p': 'distance (p)',
                'n_estimators': 'number of trees', 'hidden_layer_sizes': 'hidden layer size'}


def value_label(v):
    """Text for a hyperparameter value: None -> 'None', (32, 16) -> '32x16', 1.0 -> '1'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'None'
    if isinstance(v, tuple):
        return 'x'.join(str(int(n)) for n in v)
    return value_to_text(v)


def value_order(v):
    """Sort key: numbers ascending, text next, None last; tuples by total size."""
    if v is None:
        return (2, 0, '')
    if isinstance(v, tuple):
        return (0, sum(v), str(v))
    if isinstance(v, (int, float, np.number)):
        return (0, float(v), '')
    return (1, 0, str(v))


def format_values(values):
    return ', '.join(value_label(v) for v in values)


def parse_values(text, model_name, param):
    """Parse comma-separated search values such as '3, 5, None' or '16, 32x16' and check them."""
    kind, check, description = GRID_RULES[model_name][param]
    tokens = [t.strip() for t in str(text).split(',') if t.strip()]
    if not tokens:
        raise ValueError(f'{model_name} > {param}: enter at least one value.')
    values = []
    for token in tokens:
        try:
            if kind == 'text':
                value = token.strip('\'"')
            elif kind == 'layers':
                value = tuple(int(n) for n in token.lower().replace('(', '').replace(')', '').split('x') if n.strip())
                if not value:
                    raise ValueError
            elif kind == 'int_or_none' and token.lower() == 'none':
                value = None
            elif kind in ('int', 'int_or_none'):
                number = float(token)
                if not number.is_integer():
                    raise ValueError
                value = int(number)
            else:
                value = float(token)
                value = int(value) if value.is_integer() and value >= 1 else value
        except ValueError:
            raise ValueError(f"{model_name} > {param}: '{token}' is not valid. Each value must be {description}.")
        if not check(value):
            raise ValueError(f"{model_name} > {param}: '{token}' is not valid. Each value must be {description}.")
        if value not in values:
            values.append(value)
    return values


def build_preprocessor(numeric, categorical, missing_strategy):
    fill = {'median': 'median', 'mean': 'mean', 'mode': 'most_frequent', 'drop rows': 'median'}[missing_strategy]
    steps = []
    if numeric:
        steps.append(('num', Pipeline([('impute', SimpleImputer(strategy=fill)), ('scale', StandardScaler())]), numeric))
    if categorical:
        steps.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(steps)


def make_model(name, settings, seed):
    class_weight = 'balanced' if settings['balanced'] else None
    if name == 'Logistic Regression':
        return MonitoredLogisticRegression(max_iter=int(settings['logistic_max_iter']), class_weight=class_weight)
    if name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=seed, class_weight=class_weight)
    if name == 'K-Nearest Neighbors':
        return KNeighborsClassifier()
    if name == 'Random Forest':
        return RandomForestClassifier(random_state=seed, class_weight=class_weight)
    if name == 'Neural Network':
        # early_stopping holds back 10% of the training rows and stops when their score stops improving,
        # so the network converges instead of hitting max_iter
        return MonitoredMLP(max_iter=int(settings['mlp_max_iter']), early_stopping=True, random_state=seed)
    raise ValueError(f'Unknown model: {name}')


class ConvergenceMonitor:
    """Mixin: the model records whether it converged (in `converged_`) instead of raising a ConvergenceWarning.

    Searches run their fits in parallel worker processes, where warnings cannot be caught; the flag travels back
    with each fit's score instead (see `converged_scorer`), so non-convergence is still reported in the app.
    """
    def fit(self, X, y, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            super().fit(X, y, **kwargs)
        self.converged_ = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for w in caught:   # pass on any other warning unchanged
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn(w.message, w.category)
        return self


class MonitoredLogisticRegression(ConvergenceMonitor, LogisticRegression):
    """LogisticRegression that records convergence (see ConvergenceMonitor)."""


class MonitoredMLP(ConvergenceMonitor, MLPClassifier):
    """MLPClassifier that records convergence (see ConvergenceMonitor) and works in 32-bit floats.

    32-bit floats make training 5-25% faster (the network's matrix products dominate its run time); the chosen
    layer size and the test scores stay the same on every sample dataset.
    """
    def fit(self, X, y, **kwargs):
        return super().fit(X.astype(np.float32), y, **kwargs)

    def predict_proba(self, X):
        return super().predict_proba(X.astype(np.float32))

    def predict(self, X):
        return super().predict(X.astype(np.float32))


# models that can fail to converge, the setting that limits them, and the hyperparameter named in the note
MONITORED = {'Logistic Regression': ('logistic_max_iter', 'C'), 'Neural Network': ('mlp_max_iter', 'hidden_layer_sizes')}


def converged_scorer(estimator, X, y):
    """1 if the pipeline's model converged, else 0 (used as a second 'score' of monitored searches)."""
    return float(getattr(estimator.named_steps['model'], 'converged_', True))


def settings_from_preset(preset='Quick'):
    settings = copy.deepcopy(DEFAULTS)
    settings['preset'] = preset
    settings['models'] = {n: {'run': True, **copy.deepcopy(spec)} for n, spec in SEARCH_PRESETS[preset].items()}
    return settings


def split_data(prepared, test_size, seed):
    if not 0.05 <= float(test_size) <= 0.5:
        raise ValueError('Test size must be between 0.05 and 0.5.')
    X, y = prepared['X'], prepared['y']
    n_test = math.ceil(len(X) * float(test_size))
    if n_test < 10 or len(X) - n_test < 20:
        raise ValueError(f'Test size {test_size} leaves too few rows for training or testing ({len(X)} rows in total).')
    return train_test_split(X, y, test_size=float(test_size), random_state=int(seed), stratify=y)


# every search runs its cross-validation fits on all CPU cores at once (same results, about 1.5-2x faster on
# Colab's 2 cores). Models that can fail to converge report it through ConvergenceMonitor.
PARALLEL_SEARCH = set()   # the browser has one process


class SharedWorkSearch:
    """A randomized search with exactly the results of RandomizedSearchCV, for models where several candidates can
    share one expensive step. Candidates are grouped by every setting except `shared` ones; each (group, CV fold) is
    one job that does the expensive step once and scores all of the group's candidates. The candidates (same sampler
    and seed), folds, scores, ranking, best settings and refit are the same as RandomizedSearchCV's.
    """
    shared = ()

    def __init__(self, pipeline, param_distributions, n_iter, scoring, cv, random_state, n_jobs=-1):
        self.pipeline, self.param_distributions, self.n_iter = pipeline, param_distributions, n_iter
        self.scoring, self.cv, self.random_state, self.n_jobs = scoring, cv, random_state, n_jobs

    @staticmethod
    def _cost(group):
        return len(group)

    def fit(self, X, y):
        candidates = list(ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state))
        folds = list(check_cv(self.cv, y, classifier=True).split(X, y))
        scorer = get_scorer(self.scoring)
        groups = {}
        for i, c in enumerate(candidates):
            key = tuple(sorted((k, repr(v)) for k, v in c.items() if self.shared != 'all' and k not in self.shared))
            groups.setdefault(key, []).append((i, c))
        # biggest jobs first, so no core waits for one long job at the end
        tasks = sorted(((g, j) for g in groups.values() for j in range(len(folds))), key=lambda t: -self._cost(t[0]))
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._score_group)(self.pipeline, X, y, *folds[j], group, scorer) for group, j in tasks)
        fold_scores = np.full((len(candidates), len(folds)), np.nan)
        fold_converged = np.ones((len(candidates), len(folds)))
        for (group, j), scores in zip(tasks, results):
            for i, score, *converged in scores:   # jobs may also report whether the model converged
                fold_scores[i, j] = score
                if converged:
                    fold_converged[i, j] = converged[0]
        mean = fold_scores.mean(axis=1)
        rank = pd.Series(-mean).rank(method='min').astype(int).to_numpy()
        self.best_index_ = int(np.argmin(rank))
        self.best_params_ = candidates[self.best_index_]
        self.best_score_ = float(mean[self.best_index_])
        self.cv_results_ = {'params': candidates, 'mean_test_score': mean, 'std_test_score': fold_scores.std(axis=1),
                            'rank_test_score': rank, 'mean_test_converged': fold_converged.mean(axis=1),
                            **{f'split{j}_test_score': fold_scores[:, j] for j in range(len(folds))}}
        for name in self.param_distributions:
            self.cv_results_[f'param_{name}'] = np.array([c[name] for c in candidates], dtype=object)
        self.best_estimator_ = clone(self.pipeline).set_params(**self.best_params_).fit(X, y)
        return self


class SharedTreeForestSearch(SharedWorkSearch):
    """Random Forest search: about 20% less work, same results.

    With a fixed random_state, a forest of k trees is exactly the first k trees of a larger forest with the same
    other settings. So for each (max_depth, min_samples_leaf) group and CV fold, the largest sampled forest is built
    once, and every sampled tree count in the group is scored from its first trees.
    """
    shared = ('model__n_estimators',)

    @staticmethod
    def _cost(group):
        return max(c['model__n_estimators'] for _, c in group)

    @staticmethod
    def _score_group(pipeline, X, y, train, test, group, scorer):
        """Fit the group's largest forest on one fold, then score each candidate from its first n trees."""
        params = dict(group[0][1])
        params['model__n_estimators'] = max(c['model__n_estimators'] for _, c in group)
        fitted = clone(pipeline).set_params(**params).fit(X.iloc[train], y.iloc[train])
        forest = fitted.named_steps['model']
        all_trees = forest.estimators_
        scores = []
        for i, candidate in group:
            forest.estimators_ = all_trees[:candidate['model__n_estimators']]
            forest.n_estimators = candidate['model__n_estimators']
            scores.append((i, scorer(fitted, X.iloc[test], y.iloc[test])))
        return scores


class _FixedPredictions(ClassifierMixin, BaseEstimator):
    """Stands in for a fitted classifier whose predictions are already known, so a normal scorer can score them."""
    def __init__(self, classes, proba, labels):
        self.classes, self.proba, self.labels = classes, proba, labels
        self.classes_ = classes

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.proba

    def predict(self, X):
        return self.labels


class SharedNeighbourKNNSearch(SharedWorkSearch):
    """KNN search: the neighbours are found once per distance type and fold, same results.

    A person's neighbours depend only on the distance type p, not on K or the weighting. So for each (p, fold) the
    neighbours are found once for the largest sampled K, and each sampled (K, weights) is scored from the first K
    of them with KNeighborsClassifier's own voting rule. When people tie for the K-th place (equally distant),
    which of them counts is arbitrary and scikit-learn may break the tie differently, so those candidates are
    scored the normal way instead. That keeps every score exactly the same as RandomizedSearchCV's.
    """
    shared = ('model__n_neighbors', 'model__weights')

    @staticmethod
    def _cost(group):
        return max(c['model__n_neighbors'] for _, c in group) * len(group)

    @staticmethod
    def _score_group(pipeline, X, y, train, test, group, scorer):
        """Find the neighbours once (brute force) for the group's largest K, then score each candidate from its first K."""
        k_max = max(c['model__n_neighbors'] for _, c in group)
        fitted = clone(pipeline).set_params(model__p=group[0][1]['model__p'], model__n_neighbors=k_max,
                                            model__weights='uniform', model__algorithm='brute'
                                            ).fit(X.iloc[train], y.iloc[train])
        shared = fitted.named_steps['model']
        Z_train, y_train = shared._fit_X, y.iloc[train]
        Z = fitted.named_steps['prep'].transform(X.iloc[test])
        dist, ind = shared.kneighbors(Z, n_neighbors=min(k_max + 1, len(Z_train)))   # one extra, to detect ties
        labels = shared._y[ind]                    # class index of each neighbour
        rows = np.arange(len(Z))
        scores = []
        for i, candidate in group:
            k, weighting = candidate['model__n_neighbors'], candidate['model__weights']
            # the model RandomizedSearchCV would fit for this candidate (fitting a KNN only stores the data)
            own = clone(pipeline.named_steps['model']).set_params(n_neighbors=k, weights=weighting,
                                                                  p=candidate['model__p']).fit(Z_train, y_train)
            if own._fit_method != 'brute':
                # scikit-learn chose a tree search for this K, which breaks ties its own way: use its answer
                proba, predicted = own.predict_proba(Z), own.predict(Z)
            else:
                # the same voting as KNeighborsClassifier.predict_proba, from the shared neighbours
                weights = _get_weights(dist[:, :k], weighting)
                if weights is None:
                    weights = np.ones((len(Z), k))
                proba = np.zeros((len(Z), len(shared.classes_)))
                for j in range(k):
                    proba[rows, labels[:, j]] += weights[:, j]
                proba /= proba.sum(axis=1)[:, np.newaxis]
                predicted = shared.classes_[np.argmax(proba, axis=1)]
                # people with a tie for the K-th place (equally distant): which neighbour counts is arbitrary, so
                # use scikit-learn's own answer for just these people
                tied = (dist[:, k - 1] == dist[:, k]) if k < dist.shape[1] else np.zeros(len(Z), dtype=bool)
                if tied.any():
                    proba[tied] = own.predict_proba(Z[tied])
                    predicted[tied] = own.predict(Z[tied])
            scores.append((i, scorer(_FixedPredictions(shared.classes_, proba, predicted), Z, y.iloc[test])))
        return scores


class SharedPrepSearch(SharedWorkSearch):
    """Decision Tree and Logistic Regression search: each CV fold is preprocessed once, same results.

    Filling gaps, scaling and one-hot encoding take longer than fitting one tree or one logistic regression, and
    they do not depend on the model's settings. So each fold's data is preprocessed once and every sampled model is
    fitted on it. Whether each model converged is reported too (see ConvergenceMonitor).
    """
    shared = 'all'

    @staticmethod
    def _score_group(pipeline, X, y, train, test, group, scorer):
        """Preprocess one fold once, then fit and score every candidate's model on it."""
        prep = clone(pipeline.named_steps['prep']).fit(X.iloc[train], y.iloc[train])
        Z_train, Z_test = prep.transform(X.iloc[train]), prep.transform(X.iloc[test])
        scores = []
        for i, candidate in group:
            model = clone(pipeline.named_steps['model']).set_params(
                **{k.replace('model__', ''): v for k, v in candidate.items()}).fit(Z_train, y.iloc[train])
            scores.append((i, scorer(model, Z_test, y.iloc[test]), float(getattr(model, 'converged_', True))))
        return scores


# searches that share work between candidates (same results as RandomizedSearchCV, less work)
SHARED_SEARCH = {'Random Forest': SharedTreeForestSearch, 'K-Nearest Neighbors': SharedNeighbourKNNSearch,
                 'Decision Tree': SharedPrepSearch, 'Logistic Regression': SharedPrepSearch}


def warning_notes(caught, model_name):
    notes = []
    for w in caught:
        text = f'{model_name}: {w.category.__name__}: {str(w.message).strip()}'
        if text not in notes:
            notes.append(text)
    return notes


def train_models(prepared, settings, progress=None):
    """Split the data, tune and train the selected models, score them, and pick the best model.

    progress(fraction, text) is called before each model if given. Warnings raised while training (for example
    a model that did not converge) are returned as notes, not hidden.
    """
    seed = int(settings['random_state'])
    if settings['scoring'] not in SCORING:
        raise ValueError(f'Scoring must be one of {list(SCORING)}.')
    names = [n for n in MODEL_NAMES if settings['models'][n]['run']]
    if not names:
        raise ValueError('Select at least one model to run.')
    X_train, X_test, y_train, y_test = split_data(prepared, settings['test_size'], seed)
    results, notes = {}, []
    search_rows = PRESET_SEARCH_ROWS.get(settings.get('preset'))
    if search_rows and len(X_train) > search_rows:
        X_search, _, y_search, _ = train_test_split(X_train, y_train, train_size=search_rows, random_state=seed,
                                                    stratify=y_train)
        notes.append(f'Quick search: the settings were chosen on {search_rows:,} of the {len(X_train):,} training '
                     'rows; each best model was then trained on all of them.')
    else:
        X_search, y_search = X_train, y_train
    smallest_class = int(min(y_search.sum(), len(y_search) - y_search.sum()))

    for i, name in enumerate(names):
        if progress:
            progress(i / (len(names) + 1), f'Training {name} ({i + 1}/{len(names)})')
        spec = settings['models'][name]
        grid, cv = spec['grid'], int(spec['cv'])
        if cv < 2:
            raise ValueError(f'{name}: CV folds must be at least 2.')
        if cv > smallest_class:
            raise ValueError(f'{name}: {cv} CV folds is too many. The smaller class has only {smallest_class} '
                             f'training rows. Use at most {smallest_class} folds.')
        if name == 'K-Nearest Neighbors':
            fold_rows = len(X_search) * (cv - 1) // cv
            too_big = [k for k in grid['n_neighbors'] if k > fold_rows]
            if len(too_big) == len(grid['n_neighbors']):
                raise ValueError(f'K-Nearest Neighbors: every K is larger than {fold_rows} '
                                 '(the number of training rows in one CV fold). Use smaller values.')
            if too_big:
                grid = {**grid, 'n_neighbors': [k for k in grid['n_neighbors'] if k <= fold_rows]}
                notes.append(f'K-Nearest Neighbors: skipped K = {too_big}, larger than the {fold_rows} training '
                             'rows in one CV fold.')
        pipeline = Pipeline([
            ('prep', build_preprocessor(prepared['numeric'], prepared['categorical'], prepared['missing_strategy'])),
            ('model', make_model(name, settings, seed)),
        ])
        n_combinations = int(np.prod([len(v) for v in grid.values()]))
        monitored = name in MONITORED
        param_distributions = {f'model__{p}': v for p, v in grid.items()}
        n_iter = max(1, min(int(spec['n_iter']), n_combinations))
        n_jobs = -1 if name in PARALLEL_SEARCH else None
        if name in SHARED_SEARCH:
            search = SHARED_SEARCH[name](pipeline, param_distributions, n_iter, SCORING[settings['scoring']],
                                         cv, seed, n_jobs)
        else:
            search = RandomizedSearchCV(
                pipeline, param_distributions, n_iter=n_iter,
                # monitored models also score 'converged', so convergence is reported from every parallel fit
                scoring={'score': SCORING[settings['scoring']], 'converged': converged_scorer} if monitored
                else SCORING[settings['scoring']],
                refit='score' if monitored else True,
                cv=cv, random_state=seed, n_jobs=n_jobs,
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            search.fit(X_search, y_search)
            model = search.best_estimator_
            if X_search is not X_train:             # Quick on a big file: train the best settings on every row
                model = clone(model).fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
        notes += warning_notes(caught, name)
        if monitored:
            limit, param = MONITORED[name]
            converged = np.asarray(search.cv_results_['mean_test_converged'])
            values = [value_label(v) for v, c in zip(search.cv_results_[f'param_model__{param}'], converged) if c < 1]
            if values or not model.named_steps['model'].converged_:
                notes.append(f"{name}: ConvergenceWarning: some fits stopped at max_iter = {settings[limit]} "
                             f"before converging ({param} {values or ['of the final model']}). "
                             f'Increase {name} max_iter in the Train tab.')
        results[name] = {
            'model': model,
            'best_params': {k.replace('model__', ''): v for k, v in search.best_params_.items()},
            'cv_score': float(search.best_score_),
            'cv_results': pd.DataFrame(search.cv_results_),
            'grid': grid, 'y_prob': y_prob,
        }

    best_model = max(results, key=lambda n: results[n]['cv_score'])
    if progress:
        progress(len(names) / (len(names) + 1), 'Explaining the best model')
    return {'prepared': prepared, 'settings': copy.deepcopy(settings), 'results': results, 'notes': notes,
            'best_model': best_model, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'cache': {}}


def evaluate(y_true, y_prob, threshold=0.5):
    """Metrics when rows with probability >= threshold are predicted as the outcome."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
    }
    return metrics, confusion_matrix(y_true, y_pred, labels=[0, 1])


def comparison_table(trained, threshold=0.5):
    rows = []
    scoring = trained['settings']['scoring']
    for name, result in trained['results'].items():
        metrics, _ = evaluate(trained['y_test'], result['y_prob'], threshold)
        rows.append({'Model': name, f'CV {scoring}': result['cv_score'], **metrics,
                     'Best model': '★' if name == trained['best_model'] else ''})
    return pd.DataFrame(rows).sort_values(f'CV {scoring}', ascending=False).reset_index(drop=True)


def hyperparameter_effects(result):
    """For each searched hyperparameter: mean CV score per value tested, averaged over the other sampled settings."""
    cv_results, grid = result['cv_results'], result['grid']
    effects = {}
    for param, values in grid.items():
        column = f'param_model__{param}'
        if column not in cv_results:
            continue
        tested = [None if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in cv_results[column]]
        labels = [value_label(v) for v in tested]
        order = [value_label(v) for v in sorted(values, key=value_order)]
        unknown = set(labels) - set(order)
        if unknown:
            raise ValueError(f'{param}: search results contain values that are not in the grid: {sorted(unknown)}')
        table = (pd.DataFrame({'value': pd.Categorical(labels, categories=order, ordered=True),
                               'score': cv_results['mean_test_score']})
                 .groupby('value', observed=True)['score'].agg(['mean', 'count']).reset_index())
        table['value'] = table['value'].astype(str)
        effects[param] = table
    return effects


def transformed_columns(fitted):
    """For each column the model sees: the original column, and the category for one-hot columns."""
    prep = fitted.named_steps['prep']
    info = []
    categorical = []
    for name, _, cols in prep.transformers_:
        if name == 'cat':
            categorical = list(cols)
    for full in prep.get_feature_names_out():
        kind, rest = full.split('__', 1)
        if kind == 'num':
            info.append({'name': full, 'column': rest, 'category': None})
        else:
            column = max((c for c in categorical if rest.startswith(f'{c}_')), key=len)
            info.append({'name': full, 'column': column, 'category': rest[len(column) + 1:]})
    return info


def unscale(fitted, column, value):
    """Turn a scaled threshold of a numeric column back into the original units."""
    num = fitted.named_steps['prep'].named_transformers_['num']
    columns = list(fitted.named_steps['prep'].transformers_[0][2])
    scaler = num.named_steps['scale']
    i = columns.index(column)
    return value * scaler.scale_[i] + scaler.mean_[i]


def edge_notes(result):
    """Hyperparameters whose best value is the smallest or largest value tested (the search may be too narrow)."""
    notes = []
    for param, best in result['best_params'].items():
        if param not in EXTENDABLE or best is None or isinstance(best, str):
            continue
        tested = sorted({v for v in result['cv_results'][f'param_model__{param}']
                         if v is not None and not (isinstance(v, float) and np.isnan(v))}, key=value_order)
        has_none = any(v is None or (isinstance(v, float) and np.isnan(v))
                       for v in result['cv_results'][f'param_model__{param}'])
        if len(tested) < 2:
            continue
        floor = isinstance(best, (int, np.integer)) and best <= 1
        if value_order(best) == value_order(tested[-1]) and not has_none:
            notes.append((param, 'largest'))
        elif value_order(best) == value_order(tested[0]) and EXTENDABLE[param] == 'both' and not floor:
            notes.append((param, 'smallest'))
    return notes


MAX_SCATTER_POINTS = 2000   # scatter plots show a random sample of at most this many points


def fmt(x):
    """Readable number: 1234 -> '1,234', 0.4567 -> '0.457', 12.345 -> '12.3'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'n/a'
    x = float(x)
    if x.is_integer():
        return f'{int(x):,}'
    if abs(x) >= 100:
        return f'{x:,.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    return f'{x:.3g}'


def mark_best(fig, x, y, text, line=True, colour=BEST_COLOUR, ax=40, ay=-40):
    """Star marker + '★ ...' annotation + dashed line at the best value."""
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name='Best',
                             marker=dict(symbol='star', size=20, color=colour, line=dict(color='black', width=1))))
    fig.add_annotation(x=x, y=y, text=f'★ {text}', showarrow=True, arrowhead=2, ax=ax, ay=ay,
                       bgcolor='rgba(255,255,255,0.85)', bordercolor=colour, font=dict(size=12))
    if line:
        fig.add_hline(y=y, line_dash='dash', line_color=colour, opacity=0.8)
    return fig


def mark_note(fig, text, colour=BEST_COLOUR):
    """'★ ...' note in the corner, for charts where the best value is not a single point."""
    fig.add_annotation(text=f'★ {text}', xref='paper', yref='paper', x=0.01, y=0.99, xanchor='left', yanchor='top',
                       showarrow=False, bgcolor='rgba(255,255,255,0.85)', bordercolor=colour, font=dict(size=12))
    return fig


def _number_like(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _category_positions(fig, letter):
    """Category name -> position on the x or y axis, if that axis shows categories (else None)."""
    axis = fig.layout[f'{letter}axis']
    values = []
    for trace in fig.data:
        data = getattr(trace, letter, None) if hasattr(trace, letter) else None
        if data is not None:
            values.extend(list(data))
    if axis.type == 'category' and axis.categoryarray is not None:
        order = [str(c) for c in axis.categoryarray]
    elif axis.type == 'category' or (axis.type is None and any(isinstance(v, str) and not _number_like(v) for v in values)):
        order = list(dict.fromkeys(str(v) for v in values))
    else:
        return None
    return {c: i for i, c in enumerate(order)}


def finish(fig, title, xaxis, yaxis, height=420):
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, height=height, showlegend=True,
                      legend=dict(orientation='h', y=-0.25))
    # On a category axis Plotly reads a label's position "100" as category number 100, not the category named
    # "100", which stretches the axis; so labels on category axes are placed by the category's position instead.
    for letter in ('x', 'y'):
        positions = _category_positions(fig, letter)
        if not positions:
            continue
        for a in fig.layout.annotations:
            if a[f'{letter}ref'] in (None, letter) and a[letter] is not None and str(a[letter]) in positions:
                a[letter] = positions[str(a[letter])]
    # Plotly widens an axis until every label fits, so on a category x axis each arrow label points inwards
    # (labels on the left half extend right, labels on the right half extend left) and the axis keeps its categories.
    positions = _category_positions(fig, 'x')
    if positions:
        middle = (len(positions) - 1) / 2
        labelled = False
        for a in fig.layout.annotations:
            if a.showarrow and a.xref in (None, 'x') and isinstance(a.x, (int, float)):
                right = a.x > middle
                a.xanchor = 'right' if right else 'left'
                a.ax = -abs(a.ax or 0) if right else abs(a.ax or 0)
                labelled = True
        if labelled and fig.layout.xaxis.range is None:
            fig.update_xaxes(range=[-0.5, len(positions) - 0.5])
    return fig


# ---------------- hyperparameter effects ----------------
def graph_hyperparameter(trained, model_name, param, w):
    result = trained['results'][model_name]
    effect = hyperparameter_effects(result)[param]
    scoring = trained['settings']['scoring']
    label = PARAM_LABELS.get(param, param)
    x = list(effect['value'])
    best_i = int(effect['mean'].idxmax())
    best_x, best_y = x[best_i], float(effect['mean'][best_i])
    categorical = param in ('weights', 'p') or (model_name == 'Random Forest' and param == 'max_depth')
    fig = go.Figure()
    if categorical:
        fig.add_trace(go.Bar(x=x, y=effect['mean'], name=f'mean CV {scoring}',
                             marker_color=[BEST_COLOUR if i == best_i else BASE_COLOUR for i in range(len(x))]))
    else:
        fig.add_trace(go.Scatter(x=x, y=effect['mean'], mode='lines+markers', name=f'mean CV {scoring}',
                                 line=dict(color=BASE_COLOUR)))
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=x)
    mark_best(fig, best_x, best_y, f'Best: {param} = {best_x}, {scoring} = {best_y:.3f}')
    if categorical:
        low = float(effect['mean'].min())
        fig.update_yaxes(range=[max(0, low - 0.05), min(1, best_y + 0.03)])
    finish(fig, f'{model_name}: effect of {label}<br><sup>mean cross-validated {scoring} over the other '
                f'sampled settings</sup>', label + (' (None = no limit)' if 'None' in x else ''),
           f'Mean CV {scoring} (averaged over other settings)')

    others = effect.drop(index=best_i)
    text = [f"**Best {label}: {best_x}** (mean CV {scoring} {best_y:.3f}, from {int(effect['count'][best_i])} "
            f"sampled setting(s)), when predicting which {w['people']} {w['outcome_past']}."]
    if len(others):
        runner = others.loc[others['mean'].idxmax()]
        worst = others.loc[others['mean'].idxmin()]
        gap = best_y - float(runner['mean'])
        if len(others) == 1:
            text.append(f"It beats the other value ({runner['value']}) by {gap:.3f}.")
        else:
            text.append(f"It beats the next best value ({runner['value']}) by {gap:.3f} and the weakest "
                        f"({worst['value']}) by {best_y - float(worst['mean']):.3f}.")
        if gap < 0.002:
            text.append('The difference is tiny, so this setting hardly matters here.')
    numeric_positions = [v for v in x if v != 'None']
    if (len(numeric_positions) >= 2 and param in EXTENDABLE and not categorical and best_x in numeric_positions
            and best_x in (numeric_positions[0], numeric_positions[-1])):
        side = 'largest' if best_x == numeric_positions[-1] else 'smallest'
        is_floor = side == 'smallest' and (EXTENDABLE[param] == 'up' or best_x in ('1', '0'))
        cv = int(trained['settings']['models'][model_name]['cv'])
        fold_rows = len(trained['X_train']) * (cv - 1) // cv
        if param == 'n_neighbors' and side == 'largest' and int(float(best_x)) >= 0.5 * fold_rows:
            text.append(f'K = {best_x} already uses a large share of the {fold_rows} training rows in each CV fold, so '
                        'the search cannot usefully go higher. Such a large K means the model averages over many '
                        f'{w["people"]} at once: a sign that a smooth, simple model (such as Logistic Regression) suits '
                        'this data.')
        elif not is_floor and not (side == 'largest' and 'None' in x):
            search_best = value_label(result['best_params'][param])
            tiny = len(others) and best_y - float(others['mean'].max()) < 0.002
            if search_best == best_x and not tiny:
                more = 'larger' if side == 'largest' else 'smaller'
                preset = (' (for example the *Thorough* preset)' if trained['settings'].get('preset') == 'Quick' else '')
                text.append(f'The score was still improving at the {side} value tested, and the best combination '
                            f'found also uses it, so the best value may lie beyond it: try widening the search{preset} '
                            f'by adding {more} values.')
            else:
                why = ('the difference is negligible' if tiny else
                       f'the best combination found by the search uses {param} = {search_best}, inside the range')
                text.append(f'The average happens to be highest at the {side} value tested, but {why}, so the search '
                            'range is wide enough.')
    search_best = value_label(result['best_params'][param])
    if search_best != best_x and not any('inside the range' in t for t in text):
        text.append(f'(The single best combination found by the search uses {param} = {search_best}; the chart '
                    'shows averages over the other settings.)')
    return fig, ' '.join(text)


# ---------------- confusion matrix ----------------
def graph_confusion(trained, model_name, threshold, w):
    result = trained['results'][model_name]
    _, cm = evaluate(trained['y_test'], result['y_prob'], threshold)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    stay, out = w['stay_past'], w['outcome_past']
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=[f'Predicted: {stay}', f'Predicted: {out}'], y=[f'Actual: {stay}', f'Actual: {out}'])
    fig.add_annotation(x=f'Predicted: {out}', y=f'Actual: {out}', text='★ correctly found', showarrow=False, yshift=-28,
                       font=dict(size=12, color='black'), bgcolor='rgba(255,255,255,0.8)', bordercolor=BEST_COLOUR)
    fig.update_layout(coloraxis_showscale=False)
    finish(fig, f'{model_name}: confusion matrix (test set, threshold {threshold:.2f})', 'Predicted', 'Actual', 400)
    positives, negatives = tp + fn, tn + fp
    p = words(w)
    text = (f"Of the {positives} {w['people']} in the test set who {out}, the model correctly found **{tp}** "
            f"({pct(tp / max(positives, 1))}) and missed {fn}. Of the {negatives} who {stay}, it wrongly "
            f"{'flagged' if p['leave'] else 'predicted'} {fp} ({pct(fp / max(negatives, 1))}) as {out}. "
            f"Lowering the threshold finds more of them but raises more false alarms.")
    return fig, text


# ---------------- ROC curves ----------------
def graph_roc(trained, w, threshold=0.5, names=None):
    names = names or list(trained['results'])
    best = trained['best_model']
    y_true = trained['y_test']
    fig = go.Figure()
    for name in names:
        prob = trained['results'][name]['y_prob']
        fpr, tpr, _ = roc_curve(y_true, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC {roc_auc_score(y_true, prob):.3f})',
                                 line=dict(width=4 if name == best else 2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random guess', line=dict(dash='dash', color='grey')))
    focus = best if best in names else names[0]
    prob = trained['results'][focus]['y_prob']
    pred = prob >= threshold
    tpr_t = float(pred[np.asarray(y_true) == 1].mean())
    fpr_t = float(pred[np.asarray(y_true) == 0].mean())
    auc = roc_auc_score(y_true, prob)
    mark_best(fig, fpr_t, tpr_t, f'{focus}: AUC {auc:.3f}; threshold {threshold:.2f} here', line=False, ax=60, ay=40)
    finish(fig, 'ROC curve' + ('s of all models' if len(names) > 1 else f': {focus}') + ' (test set)',
           f"False positive rate (wrongly flagged)", f"True positive rate (found)", 460)
    text = (f"**{focus}** has an AUC of {auc:.3f}: for a random {w['person']} who {w['outcome_past']} and one who "
            f"{w['stay_past']}, it gives the first a higher score {pct(auc)} of the time (0.5 = guessing, 1 = perfect). "
            f"At the current threshold ({threshold:.2f}) it finds {pct(tpr_t)} of {w['people']} who {w['outcome_past']} "
            f"while wrongly flagging {pct(fpr_t)} of those who {w['stay_past']} (the curve's y and x values).")
    return fig, text


# ---------------- model comparison ----------------
def graph_comparison(trained, metric, threshold, w):
    table = comparison_table(trained, threshold)
    if metric not in table.columns:
        raise ValueError(f'Unknown metric: {metric}')
    shown = table.sort_values(metric).reset_index(drop=True)
    best_i = int(shown[metric].idxmax())
    best_name, best_value = shown['Model'][best_i], float(shown[metric][best_i])
    fig = go.Figure(go.Bar(x=shown[metric], y=shown['Model'], orientation='h', text=shown[metric].round(3),
                           marker_color=[BEST_COLOUR if i == best_i else BASE_COLOUR for i in range(len(shown))],
                           name=metric))
    fig.add_vline(x=best_value, line_dash='dash', line_color=BEST_COLOUR)
    fig.add_annotation(x=best_value, y=best_name, text=f'★ Best: {best_name}, {metric} = {best_value:.3f}',
                       showarrow=True, ax=-120, ay=-30, bgcolor='rgba(255,255,255,0.85)', bordercolor=BEST_COLOUR)
    finish(fig, f'Model comparison by {metric} (test set)', metric, 'Model', 380)
    fig.update_xaxes(range=[0, 1.05])

    p = words(w)
    ranked = table.sort_values(metric, ascending=False).reset_index(drop=True)
    lines = [f"**Best by {metric}: {best_name}** ({best_value:.3f})."]
    if len(ranked) > 1:
        gaps = ', '.join(f"{r['Model']} {best_value - r[metric]:.3f} behind" for _, r in ranked.iloc[1:].iterrows())
        lines.append(f'Runners-up: {gaps}.')
    chosen = trained['best_model']
    scoring = trained['settings']['scoring']
    row = table[table['Model'] == chosen].iloc[0]
    lines.append(f"The app automatically uses **{chosen}** as the best model, because it has the highest "
                 f"cross-validated {scoring} on the training data ({row[f'CV {scoring}']:.3f}); all later charts use it.")
    recall, precision = float(row['Recall']), float(row['Precision'])
    if p['leave']:
        lines.append(f"In plain words, {chosen} catches {pct(recall)} of {w['people']} who {w['outcome_past']}, but "
                     f"{one_in(1 - precision)} {w['people']} it flags as at risk would have {w['stay_past']}.")
    else:
        lines.append(f"In plain words, {chosen} finds {pct(recall)} of {w['people']} who {w['outcome_past']}, but "
                     f"{one_in(1 - precision)} {w['people']} it predicts to join {w['stay_past']}.")
    lines.append('Raising the threshold makes its predictions more precise but finds fewer of them; lowering it does the opposite.')
    return fig, ' '.join(lines)


# ---------------- feature importance of one model ----------------
def model_importance(fitted):
    """Built-in importance (trees) or |coefficients| (Logistic Regression), summed back to the original columns."""
    model = fitted.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        values, label = model.feature_importances_, 'Importance (mean impurity decrease)'
    elif hasattr(model, 'coef_'):
        values, label = np.abs(model.coef_[0]), 'Sum of |coefficients| (scaled / one-hot features)'
    else:
        return None, None
    columns = [c['column'] for c in transformed_columns(fitted)]
    return pd.Series(values, index=columns).groupby(level=0).sum().sort_values(ascending=False), label


def graph_model_importance(trained, model_name, w, top=12):
    importance, label = model_importance(trained['results'][model_name]['model'])
    if importance is None:
        fig = go.Figure()
        mark_note(fig, f'{model_name} has no built-in importance; see the {words(w)["why"]} tab (permutation importance and SHAP).')
        finish(fig, f'{model_name}: feature importance', '', '', 220)
        return fig, (f'{model_name} has no built-in feature importance. The {words(w)["why"]} tab measures the '
                     'importance of any model with permutation importance and SHAP.')
    shown = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=shown.values, y=shown.index, orientation='h', name=label,
                           marker_color=[BEST_COLOUR if i == len(shown) - 1 else BASE_COLOUR for i in range(len(shown))]))
    mark_best(fig, float(shown.values[-1]), shown.index[-1], f'Top: {shown.index[-1]}', line=False, ax=-60, ay=30)
    finish(fig, f'{model_name}: top {len(shown)} features', label, 'Feature', max(320, 26 * len(shown) + 140))
    top3 = ', '.join(importance.index[:3])
    return fig, f"{model_name} relies most on **{top3}** to tell which {w['people']} {w['outcome_past']}."


# ---------------- decision tree diagram and rules ----------------
def tree_rules(trained, model_name='Decision Tree', max_depth=3):
    """Rules (paths of the top levels) with the share of training rows that had the outcome."""
    fitted = trained['results'][model_name]['model']
    tree = fitted.named_steps['model'].tree_
    info = transformed_columns(fitted)
    Z = fitted.named_steps['prep'].transform(trained['X_train'])
    paths = fitted.named_steps['model'].decision_path(Z)
    y = np.asarray(trained['y_train'])
    counts = np.asarray(paths.sum(axis=0)).ravel()
    outcome_counts = np.asarray(paths[y == 1].sum(axis=0)).ravel()
    nodes = {}

    def describe(feature, go_left, threshold):
        col = info[feature]
        if col['category'] is None:
            value = unscale(fitted, col['column'], threshold)
            return (col['column'], 'num', ('<=' if go_left else '>'), value)
        return (col['column'], 'cat', ('not' if go_left else 'is'), col['category'])

    def walk(node, depth, conditions, parent=None, branch=None):
        stop = tree.children_left[node] == -1 or depth == max_depth
        nodes[node] = {'depth': depth, 'conditions': conditions, 'n': int(counts[node]),
                       'rate': outcome_counts[node] / max(counts[node], 1), 'parent': parent, 'branch': branch,
                       'leaf': stop, 'feature': None if stop else tree.feature[node],
                       'threshold': None if stop else tree.threshold[node], 'children': []}
        if not stop:
            for child, go_left in [(tree.children_left[node], True), (tree.children_right[node], False)]:
                nodes[node]['children'].append(child)
                walk(child, depth + 1, conditions + [describe(tree.feature[node], go_left, tree.threshold[node])],
                     node, 'yes' if go_left else 'no')
    walk(0, 0, [])
    return nodes


def rule_text(conditions):
    """Merge and phrase conditions: [('tenure','num','<=',20),('tenure','num','>',5)] -> '5 < tenure <= 20'."""
    numeric, categorical, order = {}, {}, []
    for column, kind, op, value in conditions:
        if column not in order:
            order.append(column)
        if kind == 'num':
            low, high = numeric.get(column, (None, None))
            if op == '>':
                low = value if low is None else max(low, value)
            else:
                high = value if high is None else min(high, value)
            numeric[column] = (low, high)
        else:
            entry = categorical.setdefault(column, {'is': None, 'not': []})
            if op == 'is':
                entry['is'] = value
            else:
                entry['not'].append(value)
    parts = []
    for column in order:
        if column in numeric:
            low, high = numeric[column]
            if low is not None and high is not None:
                parts.append(f'{column} between {fmt(low)} and {fmt(high)}')
            elif low is not None:
                parts.append(f'{column} above {fmt(low)}')
            else:
                parts.append(f'{column} at most {fmt(high)}')
        else:
            entry = categorical[column]
            parts.append(f"{column} = {entry['is']}" if entry['is'] is not None
                         else f"{column} not {' / '.join(entry['not'])}")
    return ' and '.join(parts)


def graph_tree(trained, w, model_name='Decision Tree', max_depth=3):
    nodes = tree_rules(trained, model_name, max_depth)
    total = nodes[0]['n']
    leaves = [k for k, v in nodes.items() if v['leaf'] and v['n'] >= max(5, 0.02 * total)]
    top_leaf = max(leaves, key=lambda k: nodes[k]['rate'])
    path = set()
    node = top_leaf
    while node is not None:
        path.add(node)
        node = nodes[node]['parent']

    # x positions: leaves left to right, parents centred over children
    x_pos, next_x = {}, [0]

    def place(k):
        if nodes[k]['leaf']:
            x_pos[k] = next_x[0]
            next_x[0] += 1
        else:
            for c in nodes[k]['children']:
                place(c)
            x_pos[k] = np.mean([x_pos[c] for c in nodes[k]['children']])
    place(0)

    fig = go.Figure()
    for k, v in nodes.items():
        for c in v['children']:
            on_path = c in path and k in path
            fig.add_trace(go.Scatter(x=[x_pos[k], x_pos[c]], y=[-v['depth'], -v['depth'] - 1], mode='lines',
                                     line=dict(color=BEST_COLOUR if on_path else '#bbb', width=4 if on_path else 1.5),
                                     hoverinfo='skip', showlegend=False))
            fig.add_annotation(x=(x_pos[k] + x_pos[c]) / 2, y=-v['depth'] - 0.5, text=nodes[c]['branch'],
                               showarrow=False, font=dict(size=10, color='#666'))
    keys = list(nodes)
    labels = []
    fitted = trained['results'][model_name]['model']
    info = transformed_columns(fitted)
    for k in keys:
        v = nodes[k]
        if v['leaf']:
            labels.append(f"{pct(v['rate'])} {w['outcome_past']}<br>n={v['n']}")
        else:
            col = info[v['feature']]
            if col['category'] is None:
                labels.append(f"{col['column']}<br>&lt;= {fmt(unscale(fitted, col['column'], v['threshold']))}?")
            else:
                labels.append(f"{col['column']}<br>not {col['category']}?")
    colours = words(w)['colours']
    fig.add_trace(go.Scatter(
        x=[x_pos[k] for k in keys], y=[-nodes[k]['depth'] for k in keys], mode='markers+text', text=labels,
        textposition='bottom center', textfont=dict(size=10), showlegend=False,
        hovertext=[f"{rule_text(nodes[k]['conditions']) or 'all ' + w['people']}<br>{pct(nodes[k]['rate'], 1)} "
                   f"{w['outcome_past']} (n={nodes[k]['n']})" for k in keys], hoverinfo='text',
        marker=dict(size=22, color=[nodes[k]['rate'] for k in keys], cmin=0, cmax=1, line=dict(color='black', width=1),
                    colorscale=[[0, colours['Low']], [0.5, colours['Medium']], [1, colours['High']]],
                    colorbar=dict(title=f"share who<br>{w['outcome_past']}"))))
    leaf = nodes[top_leaf]
    mark_best(fig, x_pos[top_leaf], -leaf['depth'], f"Highest: {pct(leaf['rate'])} {w['outcome_past']}",
              line=False, ax=0, ay=75)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-max_depth - 0.8, 0.5])
    finish(fig, f'{model_name}: top {max_depth} levels as rules<br><sup>"yes" = condition true; '
                f'colour = share who {w["outcome_past"]}; gold path = {words(w)["top_group"]}</sup>', '', '', 600)

    overall = float(np.mean(trained['y_train']))
    ranked = sorted(leaves, key=lambda k: nodes[k]['rate'], reverse=True)[:3]
    lines = [f"Overall, {pct(overall)} of {w['people']} in the training data {w['outcome_past']}. Top rules:"]
    for i, k in enumerate(ranked, 1):
        v = nodes[k]
        lines.append(f"{i}. {w['people'].capitalize()} with {rule_text(v['conditions'])}: **{pct(v['rate'])} "
                     f"{w['outcome_past']}** ({v['n']} {w['people']}, {v['rate'] / max(overall, 1e-9):.1f}x the overall rate).")
    lowest = min(leaves, key=lambda k: nodes[k]['rate'])
    lines.append(f"Lowest: {w['people']} with {rule_text(nodes[lowest]['conditions'])} ({pct(nodes[lowest]['rate'])} "
                 f"{w['outcome_past']}).")
    return fig, '\n'.join(lines)


# ---------------- outcome rate by column ----------------
def outcome_groups(values, y, max_groups=15):
    """Group a column for an outcome-rate chart: numbers into up to 5 ranges, text into its most common values."""
    values = pd.Series(values).reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    numeric = is_numeric(values)
    if numeric and values.nunique() > 8:
        bins = pd.qcut(values, q=5, duplicates='drop')
        # label each range with the smallest and largest value actually in it
        ranges = values.groupby(bins, observed=True).agg(['min', 'max'])
        names = {b: (fmt(r['min']) if r['min'] == r['max'] else f"{fmt(r['min'])} to {fmt(r['max'])}")
                 for b, r in ranges.iterrows()}
        labels = bins.map(lambda b: names.get(b, 'missing') if pd.notna(b) else 'missing').astype(str)
        order = [names[b] for b in ranges.index]
    else:
        labels = values.map(lambda v: 'missing' if pd.isna(v) else value_label(v)).astype(str)
        counts = labels.value_counts()
        order = list(counts.index[:max_groups])
        if numeric:
            order = sorted(order, key=lambda s: float(s) if s != 'missing' else np.inf)
        labels = labels.where(labels.isin(order), 'other')
        if (labels == 'other').any():
            order.append('other')
    table = pd.DataFrame({'group': labels, 'y': y}).groupby('group')['y'].agg(['mean', 'count']).reindex(order).dropna()
    return table.reset_index(), numeric


def graph_outcome_rate(df, y, column, w, title_prefix=''):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not in the data.")
    converted, _ = convert_numeric_text(df[[column]])
    table, numeric = outcome_groups(converted[column].values, y)
    overall = float(np.mean(y))
    min_n = max(10, 0.02 * len(y))
    eligible = table[table['count'] >= min_n]
    if eligible.empty:
        eligible = table
    best_i = int(eligible['mean'].idxmax())
    low_i = int(eligible['mean'].idxmin())
    p = words(w)
    fig = go.Figure(go.Bar(x=table['group'], y=table['mean'], text=[pct(v) for v in table['mean']],
                           customdata=table['count'], hovertemplate='%{x}: %{y:.1%} (n=%{customdata})',
                           marker_color=[p['highlight'] if i == best_i else BASE_COLOUR for i in range(len(table))],
                           name=f"share who {w['outcome_past']}"))
    fig.add_hline(y=overall, line_dash='dot', line_color='grey', annotation_text=f'overall {pct(overall)}')
    best = table.loc[best_i]
    mark_best(fig, best['group'], float(best['mean']), f"Highest: {column} {best['group']}, {pct(best['mean'])}",
              ax=0, ay=-45)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=list(table['group']))
    fig.update_yaxes(tickformat='.0%', range=[0, min(1, float(table['mean'].max()) * 1.35 + 0.02)])
    finish(fig, f"{title_prefix}% who {w['outcome_past']} by {column}",
           column + (' (ranges)' if numeric else ''), f"% who {w['outcome_past']}")
    fig.update_layout(showlegend=False)

    low = table.loc[low_i]
    people = w['people'].capitalize()
    in_or_eq = 'in the range' if numeric and ' to ' in str(best['group']) else '='
    text = (f"{people} with {column} {in_or_eq} **{best['group']}** {w['outcome_past']} most often: "
            f"{pct(best['mean'])} ({int(best['count'])} {w['people']}), {best['mean'] / max(overall, 1e-9):.1f}x the "
            f"overall rate of {pct(overall)}.")
    if low_i != best_i:
        ratio = best['mean'] / low['mean'] if low['mean'] > 0 else np.inf
        times = f'{ratio:.1f}x more often' if np.isfinite(ratio) else 'while none'
        text += (f" They {w['outcome_past']} {times} than {w['people']} with {column} "
                 f"{'in the range ' if numeric and ' to ' in str(low['group']) else '= '}{low['group']} "
                 f"({pct(low['mean'])}).")
    return fig, text


# ---------------- clustering ----------------
def run_clustering(source, y, columns, n_clusters, seed, n_init=10):
    """KMeans on scaled numeric columns. Labels go into a new DataFrame; the data is not changed."""
    if not columns:
        raise ValueError('Choose at least one column for clustering.')
    data, _ = convert_numeric_text(source[list(columns)])
    not_numeric = [c for c in columns if not is_numeric(data[c])]
    if not_numeric:
        raise ValueError(f'Clustering needs numeric columns. These are not numeric: {not_numeric}')
    data['__outcome__'] = np.asarray(y)
    rows = data.dropna()
    n_clusters = int(n_clusters)
    if n_clusters < 2 or n_clusters >= len(rows):
        raise ValueError(f'Number of clusters must be between 2 and {len(rows) - 1}.')
    scaled = StandardScaler().fit_transform(rows[list(columns)])
    cluster_df = rows.copy()
    cluster_df['Group'] = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=int(n_init)).fit_predict(scaled) + 1
    return cluster_df, len(data) - len(rows)


def segment_table(cluster_df, columns, w):
    overall_mean = cluster_df[list(columns)].mean()
    overall_std = cluster_df[list(columns)].std().replace(0, 1)
    rows = []
    for group, part in cluster_df.groupby('Group'):
        z = (part[list(columns)].mean() - overall_mean) / overall_std
        traits = []
        for col in z.abs().sort_values(ascending=False).index[:2]:
            traits.append(f"{'higher' if z[col] > 0 else 'lower'} {col} (avg {fmt(part[col].mean())} vs "
                          f"{fmt(overall_mean[col])})")
        rows.append({'Group': int(group), 'Size': len(part), 'Share of all': pct(len(part) / len(cluster_df)),
                     f"% who {w['outcome_past']}": round(100 * part['__outcome__'].mean(), 1),
                     'Defining traits': '; '.join(traits),
                     **{f'avg {c}': round(float(part[c].mean()), 2) for c in columns}})
    return pd.DataFrame(rows)


def graph_clusters(cluster_df, columns, x, y_col, w):
    segments = segment_table(cluster_df, columns, w)
    rate_col = f"% who {w['outcome_past']}"
    top = segments.loc[segments[rate_col].idxmax()]
    sample = cluster_df.sample(min(MAX_SCATTER_POINTS, len(cluster_df)), random_state=42)
    sample = sample.assign(Group=sample['Group'].astype(str))
    # colour each group by its outcome rate, in the mode's colours (leave: high = red; join: high = green)
    colours = words(w)['colours']
    rates = segments.set_index('Group')[rate_col]
    span = max(rates.max() - rates.min(), 1e-9)
    scale = [[0, colours['Low']], [0.5, colours['Medium']], [1, colours['High']]]
    colour_map = {str(g): px.colors.sample_colorscale(scale, [(r - rates.min()) / span])[0] for g, r in rates.items()}
    fig = px.scatter(sample, x=x, y=y_col, color='Group', opacity=0.55, color_discrete_map=colour_map,
                     category_orders={'Group': [str(g) for g in sorted(cluster_df['Group'].unique())]})
    centres = cluster_df.groupby('Group')[[x, y_col]].mean()
    fig.add_trace(go.Scatter(x=centres[x], y=centres[y_col], mode='markers+text', name='Group centres',
                             text=[f'G{g}' for g in centres.index], textposition='top center',
                             marker=dict(symbol='x', size=12, color='black')))
    tc = centres.loc[int(top['Group'])]
    mark_best(fig, float(tc[x]), float(tc[y_col]),
              f"{words(w)['top_group'].capitalize()}: Group {int(top['Group'])}, {top[rate_col]:.0f}% {w['outcome_past']}",
              line=False)
    finish(fig, f"Segments of {w['people']}: {y_col} vs {x}"
                + (f'<br><sup>showing {len(sample):,} of {len(cluster_df):,} points</sup>' if len(sample) < len(cluster_df) else ''),
           x, y_col, 480)
    lines = []
    for _, r in segments.sort_values(rate_col, ascending=False).iterrows():
        lines.append(f"- **Group {int(r['Group'])}** ({r['Size']:,} {w['people']}, {r['Share of all']}): "
                     f"{r[rate_col]:.0f}% {w['outcome_past']}; {r['Defining traits']}.")
    overall = 100 * cluster_df['__outcome__'].mean()
    lines.append(f"The {words(w)['top_group']} is **Group {int(top['Group'])}**, where {top[rate_col]:.0f}% "
                 f"{w['outcome_past']} (overall {overall:.0f}%).")
    return fig, '\n'.join(lines), segments


def elbow_curve(source, columns, max_k, seed, n_init=10):
    data, _ = convert_numeric_text(source[list(columns)])
    not_numeric = [c for c in columns if not is_numeric(data[c])]
    if not columns or not_numeric:
        raise ValueError(f'The elbow plot needs numeric columns. Not numeric: {not_numeric}')
    scaled = StandardScaler().fit_transform(data.dropna())
    ks = list(range(1, min(int(max_k), len(scaled) - 1) + 1))
    inertia = [KMeans(n_clusters=k, random_state=int(seed), n_init=int(n_init)).fit(scaled).inertia_ for k in ks]
    return pd.DataFrame({'k': ks, 'inertia': inertia})


def graph_elbow(elbow):
    k, inertia = elbow['k'].to_numpy(float), elbow['inertia'].to_numpy(float)
    if len(k) >= 3:
        # elbow = point furthest from the straight line between the first and last points
        kn = (k - k[0]) / (k[-1] - k[0])
        iv = (inertia - inertia[-1]) / max(inertia[0] - inertia[-1], 1e-9)
        best = int(np.argmax(np.abs(kn + iv - 1)))
    else:
        best = len(k) - 1
    fig = go.Figure(go.Scatter(x=k, y=inertia, mode='lines+markers', name='inertia', line=dict(color=BASE_COLOUR)))
    mark_best(fig, k[best], inertia[best], f'Suggested k = {int(k[best])}')
    finish(fig, 'Elbow plot: choose k where the curve bends', 'Number of groups k',
           'Inertia (within-group spread; lower = tighter)', 380)
    drop = (inertia[best - 1] - inertia[best]) / inertia[best - 1] if best > 0 else 0
    after = (inertia[best] - inertia[best + 1]) / inertia[best] if best + 1 < len(k) else 0
    text = (f'The curve bends most at **k = {int(k[best])}**: going to {int(k[best])} groups reduces the spread by '
            f'{pct(drop)}, while one more group only adds {pct(after)}. More groups always lower the inertia, so '
            'pick the bend rather than the minimum.')
    return fig, text


# ---------------- regression demo ----------------
def run_regression(source, target, features, test_size, seed):
    """Linear Regression of a numeric target on numeric features (the concept demo from the original notebook)."""
    if not target:
        raise ValueError('Choose a numeric target column.')
    features = [c for c in (features or []) if c != target]
    if not features:
        raise ValueError('Choose at least one feature column (other than the target).')
    data, _ = convert_numeric_text(source[features + [target]])
    not_numeric = [c for c in features + [target] if not is_numeric(data[c])]
    if not_numeric:
        raise ValueError(f'Linear Regression needs numeric columns. These are not numeric: {not_numeric}')
    rows = data.dropna()
    if len(rows) < MIN_ROWS:
        raise ValueError(f'Only {len(rows)} usable rows. The regression needs at least {MIN_ROWS}.')
    X_train, X_test, y_train, y_test = train_test_split(rows[features], rows[target], test_size=float(test_size),
                                                        random_state=int(seed))
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {'target': target, 'features': features, 'y_test': y_test, 'y_pred': y_pred,
            'r2': r2_score(y_test, y_pred), 'mae': mean_absolute_error(y_test, y_pred),
            'coefficients': pd.DataFrame({'Feature': features, 'Coefficient': model.coef_}),
            'dropped': len(data) - len(rows)}


def default_regression_columns(source, target_hint=None):
    numeric = [c for c in numeric_columns(source)]
    if not numeric:
        return None, []
    target = target_hint if target_hint in numeric else max(numeric, key=lambda c: source[c].nunique())
    return target, [c for c in numeric if c != target][:4]


def graph_regression(result):
    y_test, y_pred = np.asarray(result['y_test'], float), np.asarray(result['y_pred'], float)
    low, high = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    idx = np.random.default_rng(42).permutation(len(y_test))[:MAX_SCATTER_POINTS]
    fig = go.Figure(go.Scatter(x=y_test[idx], y=y_pred[idx], mode='markers', opacity=0.55, name='Test rows'))
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode='lines', name='Perfect prediction',
                             line=dict(dash='dash', color=BEST_COLOUR)))
    mark_note(fig, f"Best possible = dashed line; R² = {result['r2']:.3f}, MAE = {fmt(result['mae'])}")
    finish(fig, f"Regression: actual vs predicted {result['target']}", f"Actual {result['target']}",
           f"Predicted {result['target']}", 460)
    coef = result['coefficients'].reindex(result['coefficients']['Coefficient'].abs().sort_values(ascending=False).index)
    strongest = coef.iloc[0]
    quality = 'strong' if result['r2'] >= 0.7 else 'moderate' if result['r2'] >= 0.4 else 'weak'
    text = (f"Linear Regression explains {pct(result['r2'])} of the variation in {result['target']} (R² = "
            f"{result['r2']:.3f}, a {quality} fit); predictions are off by {fmt(result['mae'])} on average (MAE). "
            f"Points on the dashed line are perfect predictions. The biggest coefficient is {strongest['Feature']} "
            f"({fmt(strongest['Coefficient'])} per unit). This is a concept demo; the classification models above "
            'are the real predictions.')
    return fig, text


# ---------------- Explore tab chart builder ----------------
def graph_explore(df, kind, x=None, y=None, color=None):
    data, _ = convert_numeric_text(df)
    x, y, color = (v if v not in (None, '', '(none)') else None for v in (x, y, color))
    for col in (x, y, color):
        if col is not None and col not in data.columns:
            raise ValueError(f"Column '{col}' is not in the data.")
    if color is not None and data[color].nunique() > 30 and not is_numeric(data[color]):
        raise ValueError(f"'{color}' has too many different values to use as colour (max 30).")
    if len(data) > MAX_SCATTER_POINTS and kind == 'Scatter':
        data = data.sample(MAX_SCATTER_POINTS, random_state=42)
    if kind == 'Histogram':
        if x is None:
            raise ValueError('A histogram needs an x-axis column.')
        fig = px.histogram(data, x=x, color=color, barmode='overlay', opacity=0.75)
        if is_numeric(data[x]):
            med = float(data[x].median())
            fig.add_vline(x=med, line_dash='dash', line_color=BEST_COLOUR)
            mark_note(fig, f'Median {x} = {fmt(med)}')
            text = f'{x}: median {fmt(med)}, mean {fmt(data[x].mean())}, range {fmt(data[x].min())} to {fmt(data[x].max())}.'
        else:
            top = data[x].value_counts()
            mark_note(fig, f'Most common: {top.index[0]} ({top.iloc[0]:,})')
            text = f'Most common {x}: {top.index[0]} ({pct(top.iloc[0] / len(data))} of rows).'
        finish(fig, f'Histogram of {x}' + (f' by {color}' if color else ''), x, 'Count')
    elif kind == 'Box':
        if y is None or not is_numeric(data[y]):
            raise ValueError('A box plot needs a numeric y-axis column.')
        fig = px.box(data, x=x, y=y, color=color)
        if x:
            med = data.groupby(x)[y].median()
            mark_note(fig, f'Highest median {y}: {x} = {med.idxmax()} ({fmt(med.max())})')
            text = f'Median {y} is highest for {x} = {med.idxmax()} ({fmt(med.max())}) and lowest for {med.idxmin()} ({fmt(med.min())}).'
        else:
            mark_note(fig, f'Median {y} = {fmt(data[y].median())}')
            text = f'Median {y} is {fmt(data[y].median())}; half of the rows lie between {fmt(data[y].quantile(.25))} and {fmt(data[y].quantile(.75))}.'
        finish(fig, f'Box plot of {y}' + (f' by {x}' if x else ''), x or '', y)
    elif kind == 'Scatter':
        if x is None or y is None:
            raise ValueError('A scatter plot needs both an x-axis and a y-axis column.')
        fig = px.scatter(data, x=x, y=y, color=color, opacity=0.6)
        if is_numeric(data[x]) and is_numeric(data[y]):
            r = data[[x, y]].corr().iloc[0, 1]
            mark_note(fig, f'Correlation r = {r:.2f}')
            strength = 'strong' if abs(r) >= 0.6 else 'moderate' if abs(r) >= 0.3 else 'weak'
            text = f"{x} and {y} have a {strength} {'positive' if r > 0 else 'negative'} correlation (r = {r:.2f})."
        else:
            mark_note(fig, 'Correlation needs two numeric columns')
            text = 'Pick two numeric columns to see their correlation.'
        finish(fig, f'Scatter: {y} vs {x}', x, y)
    elif kind == 'Bar':
        if x is None:
            raise ValueError('A bar chart needs an x-axis column.')
        if data[x].nunique() > 50:
            raise ValueError(f"'{x}' has {data[x].nunique()} different values; a bar chart can show at most 50.")
        keys = list(dict.fromkeys([x] + ([color] if color else [])))
        if y is not None:
            if not is_numeric(data[y]):
                raise ValueError('For a bar chart the y-axis column must be numeric (the bars show its mean).')
            grouped = data.groupby(keys, dropna=False)[y].mean().reset_index()
            value, ylabel = y, f'Mean {y}'
        else:
            grouped = data.groupby(keys, dropna=False).size().reset_index(name='Count')
            value, ylabel = 'Count', 'Count'
        fig = px.bar(grouped, x=x, y=value, color=color, barmode='group')
        fig.update_xaxes(type='category')
        top = grouped.loc[grouped[value].idxmax()]
        mark_best(fig, str(top[x]), float(top[value]), f'Highest: {x} = {top[x]} ({fmt(top[value])})', ax=0)
        finish(fig, f'Bar chart: {ylabel} by {x}', x, ylabel)
        text = f'The highest bar is {x} = {top[x]} ({ylabel.lower()} {fmt(top[value])}).'
    elif kind == 'Correlation heatmap':
        numeric = [c for c in data.columns if is_numeric(data[c])]
        if len(numeric) < 2:
            raise ValueError('The correlation heatmap needs at least two numeric columns.')
        corr = data[numeric].corr().round(2)
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale='RdBu_r')
        pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        a, b = pairs.abs().idxmax()
        fig.add_annotation(x=b, y=a, text='★', showarrow=False, font=dict(size=22, color=BEST_COLOUR))
        mark_note(fig, f'Strongest: {a} & {b} (r = {pairs[(a, b)]:.2f})')
        finish(fig, 'Correlation between numeric columns', 'Column', 'Column', max(420, 40 * len(numeric) + 160))
        text = f'The strongest correlation is between {a} and {b} (r = {pairs[(a, b)]:.2f}).'
    else:
        raise ValueError(f'Unknown chart type: {kind}')
    return fig, text


SHAP_ROWS = 150        # rows explained for the summary chart (fewer for slow models: see SHAP_SECONDS)
SHAP_BACKGROUND = 40   # background rows for model-agnostic SHAP
SHAP_SECONDS = 20      # rough time budget for the summary chart's explanations in the browser


def _column_groups(fitted):
    """Original columns, and the indices of each one's transformed (scaled / one-hot) columns."""
    groups = {}
    for i, c in enumerate(transformed_columns(fitted)):
        groups.setdefault(c['column'], []).append(i)
    return list(groups), [np.array(v) for v in groups.values()]


def _tree_contributions(model, Z):
    """Tree path contributions: along each person's path through a tree, the change in the predicted probability at
    each split is credited to that split's column. Base value + contributions = the model's probability exactly.
    A fast way to explain tree models in the browser (the shap package is not available there)."""
    from scipy import sparse
    trees = list(getattr(model, 'estimators_', [model]))
    values = np.zeros(Z.shape)
    base = 0.0
    for tree in trees:
        t = tree.tree_
        v = t.value[:, 0, :]
        p1 = v[:, 1] / v.sum(axis=1)             # probability of the outcome at each node
        parent = np.full(t.node_count, -1)
        for children in (t.children_left, t.children_right):
            inner = children >= 0
            parent[children[inner]] = np.nonzero(inner)[0]
        child = np.nonzero(parent >= 0)[0]
        credit = sparse.csr_matrix((p1[child] - p1[parent[child]], (np.arange(len(child)), t.feature[parent[child]])),
                                   shape=(len(child), Z.shape[1]))
        values += np.asarray((tree.decision_path(Z.astype(np.float32))[:, child] @ credit).todense())
        base += p1[0]
    return values / len(trees), base / len(trees)


def _permutation_shap(predict, Z, background, groups, max_evals, rng):
    """Permutation SHAP (antithetic permutations, like shap's PermutationExplainer), switching each original column's
    transformed columns together. Returns (values per row and column group, base value)."""
    m, nb, d = len(groups), len(background), Z.shape[1]
    n_perm = max(1, max_evals // (2 * m + 1))
    values = np.zeros((len(Z), m))
    base = float(np.mean(predict(background)))
    for r, x in enumerate(Z):
        orders = [rng.permutation(m) for _ in range(n_perm)]
        batch = np.empty((n_perm, 2 * m + 1, nb, d))
        for p, order in enumerate(orders):
            current = background.copy()
            batch[p, 0] = current
            for i, g in enumerate(order):        # forward: switch to the person's values one column at a time
                current[:, groups[g]] = x[groups[g]]
                batch[p, i + 1] = current
            for i, g in enumerate(order):        # backward: switch back, in the same order
                current[:, groups[g]] = background[:, groups[g]]
                batch[p, m + 1 + i] = current
        out = predict(batch.reshape(-1, d)).reshape(n_perm, 2 * m + 1, nb).mean(axis=2)
        for p, order in enumerate(orders):
            values[r, order] += np.diff(out[p, :m + 1]) - np.diff(out[p, m:])
        values[r] /= 2 * n_perm
    return values, base


def _explainer_inputs(trained, name):
    fitted = trained['results'][name]['model']
    prep, model = fitted.named_steps['prep'], fitted.named_steps['model']
    background = prep.transform(trained['X_train'].sample(min(SHAP_BACKGROUND, len(trained['X_train'])),
                                                          random_state=42))
    names, groups = _column_groups(fitted)
    return prep, model, background, names, groups


def _affordable_shap_rows(trained, model_name=None):
    """How many people the summary chart can explain within SHAP_SECONDS (all SHAP_ROWS for tree models)."""
    name = model_name or trained['best_model']
    prep, model, background, names, groups = _explainer_inputs(trained, name)
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        return SHAP_ROWS
    import time
    probe = np.repeat(background, max(1, 2000 // len(background)), axis=0)
    start = time.perf_counter()
    model.predict_proba(probe)
    per_prediction = (time.perf_counter() - start) / len(probe)
    m = len(groups)
    per_row = max(1, max(2 * m + 1, 200) // (2 * m + 1)) * (2 * m + 1) * len(background) * per_prediction
    return int(min(SHAP_ROWS, max(20, SHAP_SECONDS / max(per_row, 1e-9))))


def shap_explain(trained, X_rows, model_name=None):
    """SHAP values in probability units, one per original column. Returns (DataFrame, base value)."""
    name = model_name or trained['best_model']
    prep, model, background, names, groups = _explainer_inputs(trained, name)
    Z = prep.transform(X_rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)   # unknown categories are encoded as zeros
        if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
            raw, base = _tree_contributions(model, Z)
            values = np.column_stack([raw[:, g].sum(axis=1) for g in groups])
        else:
            values, base = _permutation_shap(lambda z: model.predict_proba(z)[:, 1], Z, background, groups,
                                             max(2 * len(groups) + 1, 200), np.random.default_rng(42))
    return pd.DataFrame(values, columns=names, index=X_rows.index), base


def drivers(trained):
    """Permutation importance and SHAP values of the best model (cached)."""
    cache = trained['cache']
    if 'drivers' not in cache:
        name = trained['best_model']
        fitted = trained['results'][name]['model']
        scoring = SCORING[trained['settings']['scoring']]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            # big test sets: shuffle-and-score a random sample of up to 1,000 rows (seconds instead of a minute)
            rows = trained['X_test'].sample(min(1000, len(trained['X_test'])), random_state=42)
            perm = permutation_importance(fitted, rows, trained['y_test'].loc[rows.index], scoring=scoring,
                                          n_repeats=5, random_state=42)
            sample = trained['X_test'].sample(min(_affordable_shap_rows(trained), len(trained['X_test'])), random_state=42)
            shap_values, base = shap_explain(trained, sample)
        trained['notes'] += warning_notes(caught, f'{name} (explanations)')
        cache['drivers'] = {
            'permutation': pd.DataFrame({'mean': perm.importances_mean, 'std': perm.importances_std},
                                        index=trained['X_test'].columns).sort_values('mean', ascending=False),
            'shap': shap_values, 'shap_rows': sample, 'base': base,
        }
    return cache['drivers']


def top_drivers(trained, n=5):
    perm = drivers(trained)['permutation']
    positive = perm[perm['mean'] > 0]
    return list((positive if len(positive) else perm).index[:n])


def direction_phrase(trained, column, w):
    """How a column pushes the outcome, from SHAP values: e.g. 'higher tenure pushes away from churning'."""
    d = drivers(trained)
    values, rows = d['shap'][column], d['shap_rows'][column]
    p = words(w)
    if column in trained['prepared']['numeric']:
        valid = rows.notna() & values.notna()
        if valid.sum() < 3 or rows[valid].nunique() < 2:
            return f'{column} matters'
        r = np.corrcoef(rows[valid].astype(float), values[valid])[0, 1]
        if np.isnan(r) or abs(r) < 0.1:
            return f'{column} matters, without a simple up/down pattern'
        return f"higher {column} pushes {'towards' if r > 0 else 'away from'} {w['outcome_ing']}"
    means = pd.DataFrame({'v': rows.astype(str), 's': values}).groupby('v')['s'].mean()
    return f"{column} = {means.idxmax()} pushes most towards {w['outcome_ing']}"


def graph_drivers(trained, w, top=12):
    perm = drivers(trained)['permutation'].head(top).iloc[::-1]
    name = trained['best_model']
    scoring = trained['settings']['scoring']
    fig = go.Figure(go.Bar(x=perm['mean'], y=perm.index, orientation='h', error_x=dict(array=perm['std']),
                           name=f'drop in {scoring}',
                           marker_color=[BEST_COLOUR if i == len(perm) - 1 else BASE_COLOUR for i in range(len(perm))]))
    mark_best(fig, float(perm['mean'].iloc[-1]), perm.index[-1], f'Top reason: {perm.index[-1]}', line=False, ax=-40, ay=30)
    p = words(w)
    finish(fig, f"{p['why']}: top reasons ({name})<br><sup>how much the test {scoring} drops when the column is "
                'shuffled (permutation importance)</sup>', f'Drop in {scoring} (higher = more important)', 'Column',
           max(340, 28 * len(perm) + 150))
    tops = top_drivers(trained)
    lines = [f"The top reasons {w['people']} {w['outcome_past']}, according to {name}, are:"]
    for i, col in enumerate(tops, 1):
        lines.append(f'{i}. **{col}**: {direction_phrase(trained, col, w)}.')
    return fig, '\n'.join(lines)


def graph_shap_summary(trained, w, top=10):
    d = drivers(trained)
    values, rows = d['shap'], d['shap_rows']
    order = values.abs().mean().sort_values(ascending=False).index[:top][::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, col in enumerate(order):
        raw = rows[col]
        if col in trained['prepared']['numeric']:
            colour = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
            hover = [f'{col} = {fmt(v)}' for v in raw]
        else:
            codes = raw.astype(str).astype('category').cat.codes
            colour = codes / max(codes.max(), 1)
            hover = [f'{col} = {v}' for v in raw]
        fig.add_trace(go.Scatter(x=values[col], y=i + rng.uniform(-0.25, 0.25, len(values)), mode='markers',
                                 marker=dict(size=6, color=colour, colorscale='Bluered', cmin=0, cmax=1, opacity=0.7,
                                             showscale=(i == len(order) - 1),
                                             colorbar=dict(title='value<br>low→high', tickvals=[])),
                                 hovertext=hover, hoverinfo='text+x', showlegend=False, name=col))
    top_col = order[-1]
    mean_abs = float(values[top_col].abs().mean())
    mark_best(fig, mean_abs, len(order) - 1, f'Strongest: {top_col}', line=False, ax=60, ay=-30)
    fig.add_vline(x=0, line_color='grey')
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=list(order))
    p = words(w)
    finish(fig, f"SHAP summary ({trained['best_model']}): how each column moves the {p['score']}<br><sup>each dot is "
                f'one {w["person"]}; right = pushes {p["pushes"]}; colour = the column\'s value</sup>',
           f"Change in probability of having {w['outcome_past']}", '', max(380, 34 * len(order) + 160))
    text = (f"Each dot is one {w['person']}. Dots to the right raise the {p['score']}, dots to the left lower it. "
            f"**{top_col}** moves it the most (on average ±{pct(mean_abs, 1)}): {direction_phrase(trained, top_col, w)}.")
    return fig, text


def graph_directions(trained, w, n=3):
    """Outcome rate across the values of the top drivers (direction of effect)."""
    source, y = trained['prepared']['source'], trained['prepared']['y']
    return [graph_outcome_rate(source, y, col, w, title_prefix='Driver: ') for col in top_drivers(trained, n)]


# ---------------- individuals ----------------
def level_of(prob, cutoffs):
    low, high = cutoffs
    return 'High' if prob >= high else 'Medium' if prob >= low else 'Low'


def rule_matches(rule, value):
    cond, target = str(rule['Condition']).strip(), str(rule['Value']).strip()
    if cond == 'any':
        return True
    if pd.isna(value):
        return False
    try:
        a, b = float(value), float(target)
        return {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b, '==': a == b, '!=': a != b}[cond]
    except (ValueError, KeyError):
        a = value_label(value)
        return {'==': a == target, '!=': a != target}.get(cond, False)


def suggestions_for(columns, row, rules, w):
    """Suggestions from the rule table for the given columns (the person's or group's main reasons)."""
    found = []
    rules = rules if rules is not None else pd.DataFrame(columns=RULE_COLUMNS)
    for col in columns:
        for _, rule in rules.iterrows():
            feature = str(rule['Feature']).strip()
            if feature not in (col, '*'):
                continue
            value = row.get(col, np.nan) if row is not None else np.nan
            if feature == '*' or row is None or rule_matches(rule, value):
                text = str(rule['Suggestion']).format(feature=col, **w) if '{' in str(rule['Suggestion']) else str(rule['Suggestion'])
                if text not in found:
                    found.append(text)
    return found


def explain_person(trained, row, w, cutoffs=DEFAULTS['level_cutoffs'], rules=None, top=5):
    """Probability, level, waterfall chart, plain-English paragraph and suggested actions for one person."""
    X, notes = clean_new_rows(trained['prepared'], row.iloc[[0]])
    name = trained['best_model']
    prob = float(trained['results'][name]['model'].predict_proba(X)[:, 1][0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)   # unknown categories are encoded as zeros
        values, base = shap_explain(trained, X)
    contrib = values.iloc[0].sort_values(key=np.abs, ascending=False)
    p = words(w)
    level = level_of(prob, cutoffs)
    shown = contrib.head(top)
    rest = float(contrib.iloc[top:].sum())
    labels = [f'{c} = {value_label(X.iloc[0][c]) if c in trained["prepared"]["categorical"] else fmt(X.iloc[0][c])}'
              for c in shown.index]
    fig = go.Figure(go.Waterfall(
        orientation='h', measure=['absolute'] + ['relative'] * (len(shown) + 1) + ['total'],
        y=['Average'] + labels + ['All other columns', 'This person'],
        x=[base] + list(shown.values) + [rest, 0],
        increasing=dict(marker=dict(color=p['colours']['High'])),
        decreasing=dict(marker=dict(color=p['colours']['Low'] if p['leave'] else '#d62728')),
        totals=dict(marker=dict(color=p['colours'][level])), name='contribution'))
    fig.update_yaxes(autorange='reversed')
    biggest = labels[0]
    fig.add_annotation(x=0, y=1.02, xref='paper', yref='paper', xanchor='left', yanchor='bottom', showarrow=False,
                       text=f'★ {p["Score"]}: {pct(prob)} ({level}); biggest reason: {biggest}',
                       bgcolor='rgba(255,255,255,0.85)', bordercolor=BEST_COLOUR)
    finish(fig, f"Why this {w['person']}? ({name})", f"Probability of having {w['outcome_past']}", '',
           max(380, 34 * (len(shown) + 3) + 170))
    fig.update_layout(margin=dict(t=110))
    fig.update_xaxes(tickformat='.0%')
    fig.update_layout(showlegend=False)

    toward = [l for l, v in zip(labels, shown.values) if v > 0]
    away = [l for l, v in zip(labels, shown.values) if v < 0]
    # reasons to act on: what raises the risk (leave) or what holds back the likelihood (join), from all columns
    sign = 1 if p['leave'] else -1
    act_on = [c for c, v in (sign * contrib).sort_values(ascending=False).items() if v > 0][:top]
    actions = suggestions_for(act_on, X.iloc[0], rules, w)
    paragraph = (f"This {w['person']} has a {p['score']} of **{pct(prob)}** (**{level}**; the model's starting point, "
                 f"before looking at this {w['person']}'s details, is {pct(base)}). ")
    if toward:
        paragraph += f"Pushing it up: {', '.join(toward)}. "
    if away:
        paragraph += f"Pulling it down: {', '.join(away)}. "
    paragraph += (f"Suggested {p['actions']} actions to discuss: " + ' '.join(f'({i}) {a}' for i, a in enumerate(actions, 1))
                  if actions else f'No {p["actions"]} rule matches this {w["person"]}; edit the rule table to add some.')
    reasons = pd.DataFrame({'Reason': labels, 'Effect on probability': [f'{v:+.1%}' for v in shown.values]})
    return {'probability': prob, 'level': level, 'figure': fig, 'text': paragraph, 'reasons': reasons,
            'actions': actions, 'notes': notes}


def what_if(trained, original_row, changed_row, w, cutoffs=DEFAULTS['level_cutoffs']):
    model = trained['results'][trained['best_model']]['model']
    probs = []
    for row in (original_row, changed_row):
        X, _ = clean_new_rows(trained['prepared'], row.iloc[[0]])
        probs.append(float(model.predict_proba(X)[:, 1][0]))
    changes = [f'{c}: {original_row.iloc[0][c]} → {changed_row.iloc[0][c]}' for c in trained['prepared']['features']
               if str(original_row.iloc[0][c]) != str(changed_row.iloc[0][c])]
    p = words(w)
    diff = probs[1] - probs[0]
    text = (f"{p['Score']}: {pct(probs[0])} ({level_of(probs[0], cutoffs)}) → **{pct(probs[1])}** "
            f"({level_of(probs[1], cutoffs)}), a change of {100 * diff:+.1f} points"
            + (f" after changing {'; '.join(changes)}." if changes else ' (nothing was changed).'))
    return probs[1], text


def rank_people(trained, rows, w, cutoffs=DEFAULTS['level_cutoffs'], id_column=None, max_explained=300):
    """Ranked list (most at risk / most likely to join first) with each person's level and top reason."""
    X, notes = clean_new_rows(trained['prepared'], rows)
    name = trained['best_model']
    prob = trained['results'][name]['model'].predict_proba(X)[:, 1]
    p = words(w)
    out = pd.DataFrame(index=rows.index)
    if id_column and id_column in rows.columns:
        out[id_column] = rows[id_column]
    out[p['Score']] = np.round(prob, 4)
    out['Level'] = [level_of(v, cutoffs) for v in prob]
    order = np.argsort(-prob)
    explain_idx = rows.index[order[:max_explained]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        values, _ = shap_explain(trained, X.loc[explain_idx])
    sign = 1 if p['leave'] else -1      # leave: what raises the risk; join: what holds back the likelihood
    reason_col = 'Top reason' if p['leave'] else 'Main thing holding them back'
    out[reason_col] = ''
    for idx in explain_idx:
        col = (sign * values.loc[idx]).idxmax()
        out.loc[idx, reason_col] = f'{col} = {value_label(X.loc[idx, col])}'
    if len(rows) > max_explained:
        notes.append(f'Reasons are shown for the top {max_explained} {w["people"]} only (to keep it fast).')
    out = out.iloc[order].reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    for col in rows.columns:
        if col not in out.columns:
            out[col] = rows.iloc[order][col].values
    return out, notes


def sensitive_groups(source, column):
    values = source[column]
    if is_numeric(values) and values.nunique() > 6:
        bins = pd.qcut(values, q=4, duplicates='drop')
        ranges = values.groupby(bins, observed=True).agg(['min', 'max'])
        names = {b: f"{fmt(r['min'])} to {fmt(r['max'])}" for b, r in ranges.iterrows()}
        return bins.map(lambda b: names.get(b, 'missing') if pd.notna(b) else 'missing').astype(str)
    return values.map(lambda v: 'missing' if pd.isna(v) else value_label(v)).astype(str)


def fairness_check(trained, threshold, w):
    prepared = trained['prepared']
    columns = prepared['sensitive']
    p = words(w)
    if not columns:
        return pd.DataFrame(), [], 'No sensitive columns (gender, age, race/ethnicity, religion, disability, marital ' \
                                   'status, nationality) were found, so there is nothing to compare.'
    model = trained['results'][trained['best_model']]['model']
    X_test, y_test = trained['X_test'], np.asarray(trained['y_test'])
    prob = model.predict_proba(X_test)[:, 1]
    flagged = prob >= threshold
    source = prepared['source'].loc[X_test.index]
    rows, warnings_list = [], []
    for col in columns:
        groups = sensitive_groups(source, col)
        for g in sorted(groups.unique()):
            m = (groups == g).to_numpy()
            pos = m & (y_test == 1)
            rows.append({'Column': col, 'Group': g, 'People': int(m.sum()),
                         f"Actual % who {w['outcome_past']}": round(100 * y_test[m].mean(), 1),
                         f"Mean predicted %": round(100 * prob[m].mean(), 1),
                         f"% {p['flagged']}": round(100 * flagged[m].mean(), 1),
                         'Recall %': round(100 * flagged[pos].mean(), 1) if pos.sum() else np.nan})
    table = pd.DataFrame(rows)
    flag_col = f"% {p['flagged']}"
    for col, part in table.groupby('Column'):
        part = part[part['People'] >= 10]
        if len(part) < 2:
            continue
        hi, lo = part[flag_col].max(), part[flag_col].min()
        if hi > 0 and lo / hi < 0.8:
            warnings_list.append(f"{col}: {part.loc[part[flag_col].idxmin(), 'Group']} is {p['flagged']} at "
                                 f"{lo:.0f}% vs {hi:.0f}% for {part.loc[part[flag_col].idxmax(), 'Group']} "
                                 f"(ratio {lo / hi:.2f}, below 0.8).")
        rec = part['Recall %'].dropna()
        if len(rec) >= 2 and rec.max() - rec.min() > 10:
            warnings_list.append(f"{col}: recall ranges from {rec.min():.0f}% to {rec.max():.0f}% "
                                 f"({rec.max() - rec.min():.0f} points apart).")
    excluded = prepared['excluded']
    text = [f"Fairness check of **{trained['best_model']}** on the test set (threshold {threshold:.2f}). "
            + (f'The sensitive columns {excluded} were left out of training, but other columns can still act as '
               'proxies, so check the groups anyway.' if excluded else
               'Sensitive columns were used for training; consider leaving them out (Setup tab).')]
    if warnings_list:
        text.append('**Warnings:** ' + ' '.join(warnings_list))
        if not p['leave']:
            text.append('This matters especially for joining decisions such as admissions or hiring: review the '
                        'differences before using the predictions.')
    else:
        text.append('No large differences were found between groups (all flagged-rate ratios ≥ 0.8 and recall gaps ≤ 10 points).')
    return table, warnings_list, ' '.join(text)


def graph_fairness(table, w):
    p = words(w)
    if table.empty:
        fig = go.Figure()
        mark_note(fig, 'No sensitive columns found')
        return finish(fig, 'Fairness check', '', '', 220)
    labels = table['Column'] + ': ' + table['Group']
    fig = go.Figure()
    for metric, colour in [('Mean predicted %', BASE_COLOUR), (f"% {p['flagged']}", p['highlight']), ('Recall %', '#7f7f7f')]:
        fig.add_trace(go.Bar(x=labels, y=table[metric], name=metric, marker_color=colour))
    gaps = table.groupby('Column')[f"% {p['flagged']}"].agg(lambda s: s.max() - s.min())
    worst = gaps.idxmax()
    part = table[table['Column'] == worst]
    lo = part.loc[part[f"% {p['flagged']}"].idxmin()]
    fig.add_annotation(x=f"{worst}: {lo['Group']}", y=float(lo[f"% {p['flagged']}"]), showarrow=True, ay=-50,
                       text=f"★ Largest gap: {worst} ({gaps[worst]:.0f} points)", bgcolor='rgba(255,255,255,0.85)',
                       bordercolor=BEST_COLOUR)
    fig.update_layout(barmode='group')
    return finish(fig, f"Fairness check: predictions by group ({p['score']})", 'Group', 'Percent', 460)


def overall_summary(trained, w, threshold, rules, segments=None):
    p = words(w)
    name = trained['best_model']
    result = trained['results'][name]
    metrics, _ = evaluate(trained['y_test'], result['y_prob'], threshold)
    tops = top_drivers(trained, 5)
    lines = [f"### Summary: {w['people']} who {w['outcome_past']}",
             f"- **Best model:** {name} with {', '.join(f'{k} = {value_label(v)}' for k, v in result['best_params'].items())} "
             f"(CV {trained['settings']['scoring']} {result['cv_score']:.3f}; test ROC-AUC {metrics['ROC-AUC']:.3f}, "
             f"recall {pct(metrics['Recall'])}, precision {pct(metrics['Precision'])} at threshold {threshold:.2f}).",
             '- **Top 5 reasons:** ' + '; '.join(f'{i}. {c} ({direction_phrase(trained, c, w)})' for i, c in enumerate(tops, 1)) + '.']
    if segments is not None and len(segments):
        rate_col = f"% who {w['outcome_past']}"
        top = segments.loc[segments[rate_col].idxmax()]
        lines.append(f"- **{p['top_group'].capitalize()}:** Group {int(top['Group'])} ({top['Size']:,} {w['people']}, "
                     f"{top[rate_col]:.0f}% {w['outcome_past']}): {top['Defining traits']}.")
    else:
        source, y = trained['prepared']['source'], trained['prepared']['y']
        table, _ = outcome_groups(convert_numeric_text(source[[tops[0]]])[0][tops[0]].values, y)
        best = table.loc[table['mean'].idxmax()]
        lines.append(f"- **{p['top_group'].capitalize()}:** {w['people']} with {tops[0]} {best['group']} "
                     f"({pct(best['mean'])} {w['outcome_past']}).")
    actions = suggestions_for(tops, None, rules, w)
    if actions:
        lines.append(f"- **Suggested {p['actions']} actions to discuss:** " + ' '.join(f'({i}) {a}' for i, a in enumerate(actions[:6], 1)))
    lines.append('- *These are patterns in past data, not certainties. Use them to start a conversation, never as '
                 'the only basis for a decision about a person.*')
    return '\n'.join(lines)


def full_analysis(trained, w, threshold=0.5, metric='ROC-AUC', rules=None, cluster_columns=None, n_clusters=3,
                  regression=None, regression_data=None):
    """Every graph with its analysis, in the order of the original notebook. Returns a list of sections.

    regression = (target, features) for the regression demo, taken from regression_data (default: the cleaned data).
    """
    sections = []
    add = lambda title, fig, text: sections.append({'title': title, 'figure': fig, 'text': text})
    for name in trained['results']:
        for param in trained['results'][name]['grid']:
            if len(trained['results'][name]['grid'][param]) > 1:
                fig, text = graph_hyperparameter(trained, name, param, w)
                add(f'{name}: effect of {PARAM_LABELS.get(param, param)}', fig, text)
        if name == 'Decision Tree':
            fig, text = graph_tree(trained, w)
            add('Decision tree rules', fig, text)
        fig, text = graph_confusion(trained, name, threshold, w)
        add(f'{name}: confusion matrix', fig, text)
    fig, text = graph_comparison(trained, metric, threshold, w)
    add('Model comparison', fig, text)
    fig, text = graph_roc(trained, w, threshold)
    add('ROC curves', fig, text)
    p = words(w)
    fig, text = graph_drivers(trained, w)
    add(f"{p['why']}: top reasons", fig, text)
    fig, text = graph_shap_summary(trained, w)
    add('SHAP summary', fig, text)
    for fig, text in graph_directions(trained, w):
        add('Direction of effect', fig, text)
    segments = None
    source, y = trained['prepared']['source'], trained['prepared']['y']
    numeric = [c for c in trained['prepared']['numeric']]
    cluster_columns = [c for c in (cluster_columns or top_numeric(trained, 3)) if c in numeric]
    if len(cluster_columns) >= 1:
        seed = trained['settings']['random_state']
        fig, text = graph_elbow(elbow_curve(source, cluster_columns, DEFAULTS['clustering']['max_k'], seed))
        add('Elbow plot', fig, text)
        cluster_df, _ = run_clustering(source, y, cluster_columns, n_clusters, seed)
        fig, text, segments = graph_clusters(cluster_df, cluster_columns, cluster_columns[0],
                                             cluster_columns[min(1, len(cluster_columns) - 1)], w)
        add('Segments', fig, text)
    table, _, text = fairness_check(trained, threshold, w)
    add('Fairness check', graph_fairness(table, w), text)
    if regression:
        reg = run_regression(source if regression_data is None else regression_data, regression[0], regression[1],
                             trained['settings']['test_size'], trained['settings']['random_state'])
        fig, text = graph_regression(reg)
        add('Regression (concept demo)', fig, text)
    summary = overall_summary(trained, w, threshold, rules, segments)
    return sections, summary


def top_numeric(trained, n=3):
    """The most important numeric columns (for default clustering)."""
    numeric = trained['prepared']['numeric']
    ordered = [c for c in drivers(trained)['permutation'].index if c in numeric]
    return ordered[:n]


PNG_EXPORT = {'works': None}   # PNG needs Kaleido + Chrome; tried once, then skipped if it fails


def export_figures(figures, prefix):
    """Save figures as PNG (when possible) or interactive HTML. Returns (paths, notes)."""
    paths, notes = [], []
    for i, (name, fig) in enumerate(figures):
        base = os.path.join(OUTPUT_DIR, safe_name(f'{prefix}_{i + 1:02d}_{name}'))
        if PNG_EXPORT['works'] is not False:
            try:
                fig.write_image(base + '.png', width=1100, height=max(500, int(fig.layout.height or 500)))
                paths.append(base + '.png')
                PNG_EXPORT['works'] = True
                continue
            except Exception:
                PNG_EXPORT['works'] = False
                notes.append('PNG export is not available here (Kaleido needs Chrome), so charts are saved as HTML.')
        fig.write_html(base + '.html', include_plotlyjs='cdn')
        paths.append(base + '.html')
    return paths, notes


def zip_files(paths, name):
    path = os.path.join(OUTPUT_DIR, safe_name(name) + '.zip')
    with zipfile.ZipFile(path, 'w') as z:
        for p in paths:
            z.write(p, os.path.basename(p))
    return path


def markdown_to_html(text):
    html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', str(text))
    html = re.sub(r'(?<!\*)\*(?!\*)(.+?)\*', r'<i>\1</i>', html)
    out, in_list = [], False
    for line in html.split('\n'):
        item = re.match(r'^\s*(?:-|\d+\.)\s+(.*)', line)
        if item and not in_list:
            out.append('<ul>')
            in_list = True
        if not item and in_list:
            out.append('</ul>')
            in_list = False
        if line.startswith('### '):
            out.append(f'<h3>{line[4:]}</h3>')
        elif item:
            out.append(f'<li>{item.group(1)}</li>')
        elif line.strip():
            out.append(f'<p>{line}</p>')
    return '\n'.join(out + (['</ul>'] if in_list else []))


def build_report(sections, summary, w, title, synthetic=False):
    """One HTML file with the summary and every graph followed by its analysis."""
    p = words(w)
    parts = [f'<html><head><meta charset="utf-8"><title>{title}</title>'
             '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
             '<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:16px;color:#222}'
             '.note{background:#fff7e0;border-left:4px solid #ffb000;padding:8px 12px}'
             '.analysis{background:#f5f7fa;padding:8px 12px;margin-bottom:28px}</style></head><body>',
             f'<h1>{title}</h1>',
             f'<p class="note">Predictions show patterns, not certainties. They should never be the only basis for a '
             f'decision about a person. Suggested {p["actions"]} actions are things to discuss, not automatic decisions.</p>']
    if synthetic:
        parts.append('<p class="note"><b>Synthetic demo data:</b> these results come from generated data and are not '
                     'real findings.</p>')
    parts.append(markdown_to_html(summary))
    for s in sections:
        parts.append(f"<h2>{s['title']}</h2>")
        parts.append(s['figure'].to_html(full_html=False, include_plotlyjs=False))
        parts.append(f"<div class='analysis'>{markdown_to_html(s['text'])}</div>")
    parts.append('</body></html>')
    path = os.path.join(OUTPUT_DIR, safe_name(f"{title}_report") + '.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    return path


# ======================================================================================================
# Browser API: everything the web page asks for. The page's worker calls api_call(name, json_args) and gets
# JSON back: text as safe HTML, charts as Plotly JSON, tables as {columns, rows}.
# ======================================================================================================
import json as _json
import base64 as _base64

NOTICE = ('**Predictions show patterns, not certainties.** They should never be the only basis for a decision '
          'about a person. Suggested actions are things to discuss, not automatic decisions.')
SYNTHETIC_NOTICE = '**Synthetic demo data:** results on this file show how the app works; they are not real findings.'
STUDENT_PRIVACY = '**Student data:** remove names and other personal identifiers before uploading.'
UPLOAD_DIR = '/uploads'
STATE = {'files': {}, 'ctx': None, 'prepared': None, 'trained': None, 'trained_key': None, 'cluster': None,
         'person': None, 'train_info': None}
PROGRESS = {'fn': None}     # set by the worker: called with (fraction, text) while training


def _progress(fraction, text):
    if PROGRESS['fn'] is not None:
        PROGRESS['fn'](float(fraction), str(text))


# ---------- conversion to JSON / HTML ----------
def _escape(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def md(text):
    """The Markdown the app writes (bold, italics, code, lists, headings) as HTML. Everything is escaped first,
    so column names and values from a file can never inject HTML."""
    html = _escape(text or '')
    html = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html)
    html = re.sub(r'(?<![\*\w])\*(?![\*\s])([^*\n]+?)(?<!\s)\*(?![\*\w])', r'<i>\1</i>', html)
    out, open_list = [], None
    for line in html.split('\n'):
        bullet = re.match(r'^\s*[-•]\s+(.*)', line)
        number = re.match(r'^\s*\d+\.\s+(.*)', line)
        kind = 'ul' if bullet else 'ol' if number else None
        if open_list and kind != open_list:
            out.append(f'</{open_list}>')
            open_list = None
        if kind and not open_list:
            out.append(f'<{kind}>')
            open_list = kind
        heading = re.match(r'^(#{1,4})\s+(.*)', line)
        if kind:
            out.append(f'<li>{(bullet or number).group(1)}</li>')
        elif heading:
            level = min(len(heading.group(1)) + 1, 5)
            out.append(f'<h{level}>{heading.group(2)}</h{level}>')
        elif line.strip() in ('---', '***'):
            out.append('<hr>')
        elif line.strip().startswith('&gt;'):
            out.append(f'<blockquote>{line.strip()[4:].strip()}</blockquote>')
        elif line.strip() and not line.strip().startswith('&lt;!--'):
            out.append(f'<p>{line}</p>')
    if open_list:
        out.append(f'</{open_list}>')
    return '\n'.join(out)


def _plain(v):
    """One value as JSON (numbers stay numbers, missing becomes null, the rest becomes text)."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, tuple):
        return value_label(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _plain(obj)


def _table(df, max_rows=None):
    if df is None:
        return None
    df = df if max_rows is None else df.head(max_rows)
    return {'columns': [str(c) for c in df.columns],
            'rows': [[_plain(v) for v in row] for row in df.itertuples(index=False)]}


def _frame(table):
    """A {columns, rows} table from the page back into a DataFrame."""
    return pd.DataFrame(table.get('rows') or [], columns=table.get('columns') or [])


def _figure(fig):
    return _json.loads(fig.to_json()) if fig is not None else None


def _block(title, fig=None, text='', table=None):
    return {'title': title, 'figure': _figure(fig), 'html': md(text), 'table': _table(table)}


# ---------- files and setup ----------
def _guess_scenario(name, df):
    lower = name.lower()
    for scenario, sc in SCENARIOS.items():
        if lower == sc['sample'].lower():
            return scenario, sc['mode']
    columns = {c.lower() for c in df.columns}
    for scenario in ['Employees', 'Customers', 'Students', 'Teachers']:
        if any(t.lower() in columns for t in SCENARIOS[scenario]['targets'][:2]) and \
                any(i.lower() in columns for i in SCENARIOS[scenario]['id_columns'][:-1]):
            return scenario, 'leave'
    if 'attrition' in columns:
        return 'Employees', 'leave'
    if 'churn' in columns:
        return 'Customers', 'leave'
    join = any(re.search(r'enrol|join|convert|accept|subscribe|signup|sign_up|bought|purchas', c) for c in columns)
    return 'Custom', 'join' if join else 'leave'


def _file_info(name):
    df = STATE['files'][name]['df']
    scenario, mode = _guess_scenario(name, df)
    return {'name': name, 'rows': int(df.shape[0]), 'columns': int(df.shape[1]), 'scenario': scenario, 'mode': mode,
            'synthetic': '_SYNTHETIC' in name, 'column_names': [str(c) for c in df.columns]}


def add_file(name, path):
    """Read an uploaded spreadsheet (already written to the page's file system)."""
    STATE['files'][name] = {'df': load_table(path), 'path': path}
    return _file_info(name)


def remove_file(name):
    STATE['files'].pop(name, None)
    if STATE['ctx'] is not None and STATE['ctx']['name'] == name:
        STATE.update(ctx=None, prepared=None, trained=None, trained_key=None, cluster=None, person=None, train_info=None)
    return {'files': list(STATE['files'])}


def _search_values(preset):
    return {name: {'cv': spec['cv'], 'n_iter': spec['n_iter'],
                   'grid': {param: format_values(values) for param, values in spec['grid'].items()}}
            for name, spec in SEARCH_PRESETS[preset].items()}


def preset_values(preset):
    return _search_values(preset)


def _wording_for(scenario, mode, wording=None):
    if scenario != 'Custom':
        return make_wording(scenario)
    return make_wording('Custom', mode, **(wording or {}))


def _custom_defaults(name, mode):
    """Wording and starter rules for the Custom scenario (the applicants sample keeps its own)."""
    if name.startswith('custom_join') and mode == 'join':
        return {k: SCENARIOS['Custom'][k] for k in CUSTOM_WORDING['join']}, SCENARIOS['Custom']['rules']
    return dict(CUSTOM_WORDING[mode]), STARTER_RULES[mode]


def custom_mode(name, mode):
    wording, rules = _custom_defaults(name or '', mode)
    return {'wording': wording, 'rules': _table(rules)}


def select_data(name, scenario=None, mode=None):
    """Use one uploaded file: guesses the scenario, target, columns and rules, and returns everything the Setup tab
    shows (like the app's Setup tab after loading a file). Clears any trained models."""
    STATE.update(ctx=None, prepared=None, trained=None, trained_key=None, cluster=None, person=None, train_info=None)
    return _setup_payload(name, scenario, mode)


def setup_info():
    """The Setup tab's information for the file in use (nothing is reset)."""
    ctx = STATE['ctx']
    if ctx is None:
        return None
    keep = {k: STATE[k] for k in ['ctx', 'prepared', 'trained', 'trained_key', 'cluster', 'person', 'train_info']}
    payload = _setup_payload(ctx['name'], ctx['scenario'], ctx['wording']['mode'])
    STATE.update(keep)
    return payload


def _setup_payload(name, scenario=None, mode=None):
    if name not in STATE['files']:
        raise ValueError(f"'{name}' is not uploaded.")
    df = STATE['files'][name]['df']
    guessed, guessed_mode = _guess_scenario(name, df)
    scenario = scenario or guessed
    mode = mode or (guessed_mode if scenario == 'Custom' else SCENARIOS[scenario]['mode'])
    columns = [str(c) for c in df.columns]
    target = guess_target(df, scenario)
    drop = [c for c in guess_id_columns(df, scenario) if c != target]
    features = [c for c in columns if c != target and c not in drop]
    sensitive = [c for c in find_sensitive_columns(df, scenario) if c != target]
    id_column = next((c for c in drop if df[c].is_unique), None)
    numeric = numeric_columns(df)
    hint = SCENARIOS[scenario].get('regression') if scenario != 'Custom' or name.startswith('custom_join') else None
    if hint and set([hint[0], *hint[1]]) <= set(numeric):
        reg_t, reg_f = hint
    else:
        usable = [c for c in numeric if c not in drop and c != target]
        reg_t, reg_f = default_regression_columns(df[usable]) if usable else (None, [])
    if scenario == 'Custom':
        wording, rules = _custom_defaults(name, mode)
    else:
        wording, rules = {k: SCENARIOS[scenario][k] for k in CUSTOM_WORDING['join']}, SCENARIOS[scenario]['rules']
    w = _wording_for(scenario, mode, wording)
    synthetic = '_SYNTHETIC' in name
    STATE['ctx'] = {'scenario': scenario, 'df': df, 'name': name, 'synthetic': synthetic, 'id_column': id_column,
                    'wording': w}
    status = [f'**{name}**: {df.shape[0]:,} rows, {df.shape[1]} columns.']
    if synthetic:
        status.append(SYNTHETIC_NOTICE)
    personal = find_personal_columns(df)
    if scenario == 'Students':
        status.append(STUDENT_PRIVACY)
    elif personal:
        status.append(f'**Personal identifiers found:** {personal}. They are dropped by default.')
    p = words(w)
    return {
        'file': name, 'scenario': scenario, 'mode': w['mode'], 'wording': wording, 'columns': columns,
        'numeric': numeric, 'target': target, 'positive': default_positive_class(df, target),
        'classes': target_classes(df, target), 'drop': drop, 'features': features, 'sensitive': sensitive,
        'id_column': id_column, 'rules': _table(rules), 'regression': {'target': reg_t, 'features': list(reg_f)},
        'preview': _table(df.head(20)), 'column_info': _table(describe_columns(df, sensitive)),
        'status_html': md('\n\n'.join(status)),
        'info_html': md(f"**{SCENARIOS[scenario]['name']}** - mode: **{w['mode']}** ({p['score']}; "
                        f"{p['actions']} suggestions)."),
    }


def target_changed(target, drop=None):
    df = STATE['ctx']['df']
    return {'classes': target_classes(df, target), 'positive': default_positive_class(df, target),
            'features': [c for c in df.columns if c != target and c not in (drop or [])]}


# ---------- training ----------
def _settings(o):
    settings = settings_from_preset(o.get('preset', DEFAULTS['preset']))
    settings['test_size'] = float(o.get('test_size', DEFAULTS['test_size']))
    settings['random_state'] = int(o.get('seed', DEFAULTS['random_state']))
    settings['scoring'] = o.get('scoring', DEFAULTS['scoring'])
    chosen = o.get('models') or MODEL_NAMES
    for name in MODEL_NAMES:
        settings['models'][name]['run'] = name in chosen
    settings['balanced'] = bool(o.get('balanced', DEFAULTS['balanced']))
    for key in ('logistic_max_iter', 'mlp_max_iter'):
        value = int(o.get(key, DEFAULTS[key]))
        if value < 1:
            raise ValueError(f'{key} must be at least 1.')
        settings[key] = value
    for name, spec in (o.get('search') or {}).items():
        if name not in settings['models']:
            continue
        for key in ('cv', 'n_iter'):
            if key in spec:
                if int(spec[key]) < 1:
                    raise ValueError(f'{name}: {key} must be at least 1.')
                settings['models'][name][key] = int(spec[key])
        for param, text in (spec.get('grid') or {}).items():
            settings['models'][name]['grid'][param] = parse_values(text, name, param)
    return settings


def _training_key(o):
    keys = ['file', 'scenario', 'mode', 'target', 'positive', 'drop', 'features', 'sensitive', 'exclude', 'missing',
            'preset', 'test_size', 'seed', 'scoring', 'models', 'balanced', 'logistic_max_iter', 'mlp_max_iter', 'search']
    return _json.dumps({k: o.get(k) for k in keys}, sort_keys=True, default=str)


def _first_person(trained, ctx):
    model = trained['results'][trained['best_model']]['model']
    prob = model.predict_proba(trained['X_test'])[:, 1]
    row = ctx['df'].loc[[trained['X_test'].index[int(np.argmax(prob))]]]
    return row[trained['prepared']['features']]


def train(options):
    """Prepare the data and train every chosen model (the app's 'Apply and train')."""
    o = options
    if STATE['ctx'] is None or STATE['ctx']['name'] != o.get('file'):
        select_data(o['file'], o.get('scenario'), o.get('mode'))
    ctx = dict(STATE['ctx'])
    ctx['scenario'] = o.get('scenario') or ctx['scenario']
    ctx['wording'] = _wording_for(ctx['scenario'], o.get('mode') or ctx['wording']['mode'], o.get('wording'))
    STATE['ctx'] = ctx
    df, w = ctx['df'], ctx['wording']
    drop = o.get('drop') or []
    features = [c for c in (o.get('features') or []) if c not in drop]
    prepared = prepare_outcome_data(df, o['target'], features, o.get('positive'), o.get('missing', 'median'),
                                    bool(o.get('exclude', True)), o.get('sensitive') or [], w)
    settings = _settings(o)
    _progress(0.01, 'Preparing the data')
    trained = train_models(prepared, settings, progress=_progress)
    _progress(0.97, 'Explaining the best model')
    drivers(trained)
    STATE.update(prepared=prepared, trained=trained, trained_key=_training_key(o), cluster=None, person=None)
    y = prepared['y']
    names = list(trained['results'])
    notes = prepared['notes'] + [n for n in trained['notes'] if 'JsProxy' not in n]
    status = (f"**Trained {len(names)} model(s)** on {len(trained['y_train']):,} {w['people']}, tested on "
              f"{len(trained['y_test']):,} ({len(prepared['features'])} features; {pct(y.mean(), 1)} "
              f"{w['outcome_past']}). Best model: **{trained['best_model']}** (highest CV {settings['scoring']}). "
              'All tabs use it.')
    p = words(w)
    columns = top_numeric(trained, 3) or prepared['numeric'][:3]
    best = trained['results'][trained['best_model']]
    metrics, _ = evaluate(trained['y_test'], best['y_prob'], DEFAULTS['threshold'])
    top = top_drivers(trained, 1)
    kpis = {'people': w['people'], 'rows': int(len(y)), 'rate': pct(y.mean(), 1), 'outcome': w['outcome_past'],
            'best': trained['best_model'], 'auc': round(float(metrics['ROC-AUC']), 3),
            'high': int((np.asarray(best['y_prob']) >= DEFAULTS['level_cutoffs'][1]).sum()), 'test': int(len(trained['y_test'])),
            'top_reason': top[0] if top else '', 'top_phrase': direction_phrase(trained, top[0], w) if top else '',
            'score': p['score'], 'leave': p['leave']}
    STATE['train_info'] = {
        'status_html': md(status + ('\n\n' + '\n'.join(f'- {n}' for n in notes) if notes else '')),
        'models': names, 'best': trained['best_model'], 'numeric': prepared['numeric'],
        'features': prepared['features'], 'categorical': prepared['categorical'], 'cluster_columns': columns,
        'person': _table(_first_person(trained, ctx)), 'why_label': p['why'], 'leave': p['leave'],
        'predict_header_html': md(f"### Predict & explain: {p['score']} (best model: {trained['best_model']})\n"
                                  f"Levels: Low < Medium < High, coloured "
                                  f"{'red for High (bad)' if p['leave'] else 'green for High (good)'}. {NOTICE}"),
        'id_column': ctx['id_column'], 'kpis': kpis,
    }
    return STATE['train_info']


def trained_info():
    return STATE.get('train_info') if STATE['trained'] is not None else None


def _need():
    if STATE['trained'] is None:
        raise ValueError('Train the models first (click "Apply and train", or ask something in the prompt box).')
    return STATE['trained'], STATE['ctx']


# ---------- tab views ----------
def model_view(name, threshold=0.5):
    trained, ctx = _need()
    w = ctx['wording']
    result = trained['results'][name]
    metrics, _ = evaluate(trained['y_test'], result['y_prob'], threshold)
    params = ', '.join(f'{k} = {value_label(v)}' for k, v in result['best_params'].items())
    best = ' (★ best model)' if name == trained['best_model'] else ''
    blocks = [_block('Confusion matrix', *graph_confusion(trained, name, threshold, w)),
              _block('ROC curve', *graph_roc(trained, w, threshold, names=[name]))]
    for param, values in result['grid'].items():
        if len(values) > 1:
            blocks.append(_block(f'Effect of {param}', *graph_hyperparameter(trained, name, param, w)))
    blocks.append(_block('Feature importance', *graph_model_importance(trained, name, w)))
    if name == 'Decision Tree':
        blocks.append(_block('The decision tree as rules', *graph_tree(trained, w)))
    return {'params_html': md(f"**{name}{best}** - best settings: {params}; CV {trained['settings']['scoring']} "
                              f"{result['cv_score']:.3f}."),
            'metrics': _table(pd.DataFrame([{'Model': name, 'Threshold': threshold,
                                             **{k: round(v, 4) for k, v in metrics.items()}}])),
            'blocks': blocks}


def compare_view(metric='ROC-AUC', threshold=0.5):
    trained, ctx = _need()
    w = ctx['wording']
    return {'table': _table(comparison_table(trained, threshold).round(4)),
            'blocks': [_block('Model comparison', *graph_comparison(trained, metric, threshold, w)),
                       _block('ROC curves', *graph_roc(trained, w, threshold))]}


METHOD_NOTE = ('In the browser, explanations use **tree path contributions** for tree models (Decision Tree, '
               'Random Forest) and **permutation SHAP** for the other models. Both add up exactly to each '
               "person's predicted probability.")


def why_view():
    trained, ctx = _need()
    w = ctx['wording']
    p = words(w)
    blocks = [_block('Top reasons (permutation importance)', *graph_drivers(trained, w)),
              _block('SHAP summary', *graph_shap_summary(trained, w))]
    blocks += [_block(f'Direction of effect {i + 1}', fig, text) for i, (fig, text) in enumerate(graph_directions(trained, w))]
    return {'title_html': md(f"### {p['why']}: {w['people']} who {w['outcome_past']} (best model: {trained['best_model']})"),
            'note_html': md(METHOD_NOTE), 'blocks': blocks}


def _rules(rules):
    if rules is None:
        return SCENARIOS[STATE['ctx']['scenario']]['rules']
    return _frame(rules).fillna('')


def segments_view(columns=None, k=3, kmax=8, x=None, y=None, threshold=0.5, rules=None):
    trained, ctx = _need()
    w = ctx['wording']
    prepared = trained['prepared']
    seed = trained['settings']['random_state']
    columns = [c for c in (columns or top_numeric(trained, 3)) if c in prepared['source'].columns]
    if not columns:
        raise ValueError('Choose at least one numeric column for the segments.')
    elbow_fig, elbow_text = graph_elbow(elbow_curve(prepared['source'], columns, int(kmax), seed))
    cluster_df, dropped = run_clustering(prepared['source'], prepared['y'], columns, int(k), seed)
    x = x if x in columns else columns[0]
    y = y if y in columns else columns[min(1, len(columns) - 1)]
    c_fig, c_text, segments = graph_clusters(cluster_df, columns, x, y, w)
    if dropped:
        c_text += f'\n\n({dropped} rows with missing values were left out of the segments.)'
    STATE['cluster'] = {'df': cluster_df, 'columns': columns, 'segments': segments}
    summary = overall_summary(trained, w, threshold, _rules(rules), segments)
    return {'columns': columns, 'x': x, 'y': y,
            'blocks': [_block('Elbow plot', elbow_fig, elbow_text), _block('Segments', c_fig, c_text)],
            'table': _table(segments.drop(columns=[c for c in segments.columns if c.startswith('avg ')])),
            'summary_html': md(summary)}


def cluster_axes(x, y):
    trained, ctx = _need()
    state = STATE['cluster']
    if state is None or x not in state['columns'] or y not in state['columns']:
        raise ValueError('Update the segments first.')
    fig, text, _ = graph_clusters(state['df'], state['columns'], x, y, ctx['wording'])
    return _block('Segments', fig, text)


def fairness_view(threshold=0.5):
    trained, ctx = _need()
    table, warns, text = fairness_check(trained, threshold, ctx['wording'])
    return {'block': _block('Fairness', graph_fairness(table, ctx['wording']), text), 'table': _table(table)}


def explore(kind='Histogram', x=None, y=None, color=None):
    if STATE['ctx'] is None:
        raise ValueError('Load a dataset first.')
    none = lambda v: None if v in (None, '', '(none)') else v
    return _block(kind, *graph_explore(STATE['ctx']['df'], kind, none(x), none(y), none(color)))


def outcome_rate(column):
    trained, ctx = _need()
    prepared = trained['prepared']
    return _block(f'Outcome rate by {column}',
                  *graph_outcome_rate(ctx['df'].loc[prepared['X'].index], prepared['y'], column, ctx['wording']))


def _cutoffs(cutoffs):
    low, high = (cutoffs or DEFAULTS['level_cutoffs'])
    if float(low) >= float(high):
        raise ValueError('"Medium starts at" must be lower than "High starts at".')
    return (float(low), float(high))


def explain(row, cutoffs=None, rules=None):
    trained, ctx = _need()
    row = _frame(row)
    if row.empty:
        raise ValueError('The person form is empty.')
    result = explain_person(trained, row.iloc[[0]], ctx['wording'], _cutoffs(cutoffs), _rules(rules))
    STATE['person'] = row.iloc[[0]]
    colour = words(ctx['wording'])['colours'][result['level']]
    return {'probability': result['probability'], 'level': result['level'], 'colour': colour,
            'header_html': md(f"## {pct(result['probability'])} - {result['level']}"),
            'html': md(result['text'] + ('\n\n' + '\n'.join(f'- {n}' for n in result['notes']) if result['notes'] else '')),
            'figure': _figure(result['figure']), 'reasons': _table(result['reasons'])}


def whatif(row, cutoffs=None):
    trained, ctx = _need()
    if STATE['person'] is None:
        raise ValueError('Click "Predict and explain" first; the what-if starts from that person.')
    _, text = what_if(trained, STATE['person'], _frame(row).iloc[[0]], ctx['wording'], _cutoffs(cutoffs))
    return {'html': md(text)}


def _find_person(key):
    trained, ctx = _need()
    df, key = ctx['df'], str(key).strip()
    if ctx['id_column'] and key in set(df[ctx['id_column']].astype(str)):
        return df[df[ctx['id_column']].astype(str) == key]
    if key.isdigit() and int(key) < len(df):
        return df.iloc[[int(key)]]
    raise ValueError(f"No {ctx['wording']['person']} with ID '{key}'"
                     + (f" in column {ctx['id_column']}" if ctx['id_column'] else '') + '; or use a row number.')


def load_person(key):
    trained, ctx = _need()
    return {'row': _table(_find_person(key)[trained['prepared']['features']])}


def _explain_budget(trained):
    return int(min(300, _affordable_shap_rows(trained)))


def rank(cutoffs=None, file=None, limit=200, explain_rows=None):
    """The ranked list: the test set, or every person in an uploaded file."""
    trained, ctx = _need()
    w = ctx['wording']
    if file:
        if file not in STATE['files']:
            raise ValueError(f"'{file}' is not uploaded.")
        rows, source = STATE['files'][file]['df'], file
    else:
        rows, source = ctx['df'].loc[trained['X_test'].index], 'the test set'
    id_column = ctx['id_column'] if ctx['id_column'] in rows.columns else None
    missing = [c for c in trained['prepared']['features'] if c not in rows.columns]
    if missing:
        raise ValueError(f"{source} does not have the columns the current model uses (missing: {', '.join(missing[:8])}"
                         f"{'...' if len(missing) > 8 else ''}). To analyse it on its own, ask: analyse {source}")
    budget = _explain_budget(trained) if explain_rows is None else min(_explain_budget(trained), int(explain_rows))
    ranked, notes = rank_people(trained, rows, w, _cutoffs(cutoffs), id_column, max_explained=budget)
    name = safe_name(f"{ctx['scenario']}_{w['people']}_{words(w)['list_file']}") + '.csv'
    return {'table': _table(ranked, limit), 'csv': ranked.to_csv(index=False), 'name': name, 'source': source,
            'total': int(len(ranked)), 'notes_html': md('\n'.join(f'- {n}' for n in notes))}


def results_csv(threshold=0.5):
    trained, ctx = _need()
    w = ctx['wording']
    return {'name': safe_name(f"{ctx['scenario']}_{w['people']}_{w['outcome_past']}_model_comparison") + '.csv',
            'text': comparison_table(trained, threshold).to_csv(index=False)}


def _analysis(threshold, metric, rules, columns, k, reg_target, reg_features):
    trained, ctx = _need()
    regression = (reg_target, reg_features) if reg_target and reg_features else None
    rows = ctx['df'].loc[trained['prepared']['X'].index]
    return full_analysis(trained, ctx['wording'], threshold, metric, _rules(rules), columns or None, int(k),
                         regression, rows)


def report(threshold=0.5, metric='ROC-AUC', rules=None, columns=None, k=3, reg_target=None, reg_features=None):
    trained, ctx = _need()
    sections, summary = _analysis(threshold, metric, rules, columns, k, reg_target, reg_features)
    w = ctx['wording']
    title = f"{ctx['scenario']} - {w['people']} who {w['outcome_past']}"
    path = build_report(sections, summary, w, title, ctx['synthetic'])
    return {'name': os.path.basename(path), 'html': open(path, encoding='utf-8').read()}


def chart_list(threshold=0.5, metric='ROC-AUC', rules=None, columns=None, k=3, reg_target=None, reg_features=None):
    trained, ctx = _need()
    sections, _ = _analysis(threshold, metric, rules, columns, k, reg_target, reg_features)
    w = ctx['wording']
    prefix = f"{ctx['scenario']}_{w['people']}"
    return {'zip_name': safe_name(f"{prefix}_{w['outcome_past']}_charts") + '.zip',
            'charts': [{'name': safe_name(f"{prefix}_{i + 1:02d}_{s['title']}"), 'figure': _figure(s['figure'])}
                       for i, s in enumerate(sections)]}


ORIGINAL_GRAPHS = [r'^Logistic Regression: effect of C', r'^Decision Tree: effect of (max depth|min samples leaf)',
                   r'^Decision tree rules', r'^K-Nearest Neighbors: effect of (K|weight)',
                   r'^Random Forest: effect of (number of trees|max depth|min samples leaf)', r'^Model comparison',
                   r'^Segments', r'^Regression']


def all_graphs(threshold=0.5, metric='ROC-AUC', rules=None, columns=None, k=3, reg_target=None, reg_features=None):
    """Every graph, grouped (each model, all models, why, segments, fairness, regression). The graphs that were in
    the original notebook are tagged."""
    trained, ctx = _need()
    sections, _ = _analysis(threshold, metric, rules, columns, k, reg_target, reg_features)
    why = words(ctx['wording'])['why']
    groups = {}
    for s in sections:
        title = s['title']
        group = next((m for m in MODEL_NAMES if title.startswith(m)), None)
        if group is None:
            group = ('Decision Tree' if title.startswith('Decision tree') else
                     'All models' if title in ('Model comparison', 'ROC curves') else
                     'Segments' if title in ('Elbow plot', 'Segments') else
                     'Fairness' if title.startswith('Fairness') else
                     'Regression' if title.startswith('Regression') else why)
        block = _block(title, s['figure'], s['text'])
        if any(re.search(pattern, title) for pattern in ORIGINAL_GRAPHS):
            block['tag'] = 'Graph from the original notebook'
        groups.setdefault(group, []).append(block)
    blocks = [b for g in groups.values() for b in g]
    return {'groups': [{'name': g, 'blocks': b} for g, b in groups.items()], 'count': len(blocks),
            'original': sum('tag' in b for b in blocks)}


def make_zip(names, data):
    """Zip files sent by the page (base64 contents). Returns the zip as base64."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in zip(names, data):
            z.writestr(name, _base64.b64decode(content))
    return {'b64': _base64.b64encode(buffer.getvalue()).decode()}


def regression(target, features, test_size=0.2, seed=42):
    if STATE['ctx'] is None:
        raise ValueError('Load a dataset first.')
    ctx, prepared = STATE['ctx'], STATE['prepared']
    source = ctx['df'] if prepared is None else ctx['df'].loc[prepared['X'].index]
    result = run_regression(source, target, features or [], float(test_size), int(seed))
    fig, text = graph_regression(result)
    if result['dropped']:
        text += f" ({result['dropped']} rows with missing values were left out.)"
    return {'block': _block('Linear Regression', fig, text), 'coefficients': _table(result['coefficients'].round(4))}


# ---------- context for the optional AI answer ----------
def ai_context():
    """A compact text summary of the current results (no raw data) for the optional Gemini answer."""
    if STATE['ctx'] is None:
        names = ', '.join(STATE['files']) or 'none'
        return f'No dataset is selected yet. Uploaded files: {names}.'
    ctx = STATE['ctx']
    w, df = ctx['wording'], ctx['df']
    lines = [f"Dataset: {ctx['name']} ({df.shape[0]} rows, {df.shape[1]} columns)"
             + (' - SYNTHETIC demo data' if ctx['synthetic'] else ''),
             f"Scenario: {SCENARIOS[ctx['scenario']]['name']}; mode {w['mode']}; one person is a {w['person']}; "
             f"outcome: {w['outcome_past']}.",
             'Columns: ' + ', '.join(f'{c} ({"numeric" if c in numeric_columns(df) else "text"})' for c in df.columns)]
    trained = STATE['trained']
    if trained is None:
        lines.append('No model has been trained yet.')
        return '\n'.join(lines)
    prepared = trained['prepared']
    lines.append(f"Target: {prepared['target']} (outcome value '{prepared['positive_class']}'); "
                 f"{pct(prepared['y'].mean(), 1)} {w['outcome_past']}. Features used: {', '.join(prepared['features'])}.")
    lines.append('Model comparison (test set, threshold 0.5):\n' + comparison_table(trained, 0.5).round(3).to_string(index=False))
    lines.append(f"Best model: {trained['best_model']} (highest cross-validated {trained['settings']['scoring']}).")
    d = drivers(trained)
    lines.append('Top reasons (permutation importance, then direction): ' + '; '.join(
        f"{c} ({direction_phrase(trained, c, w)})" for c in top_drivers(trained, 5)))
    try:
        table, warns, text = fairness_check(trained, 0.5, w)
        lines.append('Fairness check: ' + re.sub(r'\s+', ' ', text)[:800])
    except Exception:
        pass
    ranked, _ = rank_people(trained, ctx['df'].loc[trained['X_test'].index], w, DEFAULTS['level_cutoffs'],
                            ctx['id_column'], max_explained=10)
    lines.append(f"Top 10 by {words(w)['score']} in the test set:\n" + ranked.head(10).iloc[:, :5].to_string(index=False))
    lines.append('Summary:\n' + overall_summary(trained, w, 0.5, SCENARIOS[ctx['scenario']]['rules']))
    return '\n'.join(lines)


def email_summary(name='', org='', link=''):
    """Plain-text results to share by email: key numbers, top reasons, the most at-risk people and suggested actions.
    No spreadsheet rows are included apart from the IDs and scores of the top people. 'short' fits in a mailto link."""
    trained, ctx = _need()
    w = ctx['wording']
    p = words(w)
    k = (STATE.get('train_info') or {}).get('kpis') or {}
    tops = top_drivers(trained, 5)
    ranked, _ = rank_people(trained, ctx['df'].loc[trained['X_test'].index], w, DEFAULTS['level_cutoffs'],
                            ctx['id_column'], max_explained=10)
    id_col = ctx['id_column'] if ctx['id_column'] in ranked.columns else None
    reason_col = 'Top reason' if p['leave'] else 'Main thing holding them back'
    people = [f"{r['Rank']}. {r[id_col] if id_col else 'Row ' + str(r['Rank'])}: {r[p['Score']]:.0%} ({r['Level']})"
              + (f" - {r[reason_col]}" if r[reason_col] else '') for _, r in ranked.head(10).iterrows()]
    actions = suggestions_for(tops, None, SCENARIOS[ctx['scenario']]['rules'], w)
    subject = f"ChurnSense results: {ctx['name']}" + (f' ({org})' if org else '')
    head = [f"ChurnSense results for {ctx['name']}" + (' (SYNTHETIC test data)' if ctx['synthetic'] else '')]
    if name or org:
        head.append('Shared by ' + ', '.join(x for x in [name, org] if x))
    key = [f"Key numbers",
           f"- {k.get('rows', len(trained['prepared']['y'])):,} {w['people']} analysed; {k.get('rate', '')} {w['outcome_past']}",
           f"- Best model: {trained['best_model']} (ROC-AUC {k.get('auc', '')})",
           f"- {k.get('high', '')} of {k.get('test', '')} in the test set are {'high risk' if p['leave'] else 'highly likely'}"]
    reasons = ['Top reasons'] + [f'{i}. {c}: {direction_phrase(trained, c, w)}' for i, c in enumerate(tops, 1)]
    title = f"{'Most at risk' if p['leave'] else 'Most likely'} (test set, {p['score']})"
    foot = ['These are patterns in past data, not certainties. Use them to start a conversation, never as the only '
            'basis for a decision about a person.'] + ([f'Made with ChurnSense: {link}'] if link else [])
    full = head + [''] + key + [''] + reasons + ['', title] + people
    if actions:
        full += ['', f"Suggested {p['actions']} actions to discuss"] + [f'- {a}' for a in actions[:6]]
    full += [''] + foot
    short = head + [''] + key + [''] + reasons[:4] + ['', title] + people[:5] + [''] + foot
    return {'subject': subject, 'text': '\n'.join(full), 'short': '\n'.join(short)}


# ---------- results as a Word document (.docx), for sharing ----------
def share_pack(cutoffs=None):
    """The charts for the Word document (the page turns them into PNG images) and the ranked list as a CSV file."""
    trained, ctx = _need()
    w = ctx['wording']
    charts = [('Top reasons', *graph_drivers(trained, w)),
              ('Model comparison', *graph_comparison(trained, 'ROC-AUC', DEFAULTS['threshold'], w))]
    charts += [(f'Direction of effect {i + 1}', fig, text) for i, (fig, text) in enumerate(graph_directions(trained, w, 2))]
    charts.append(('Confusion matrix of the best model',
                   *graph_confusion(trained, trained['best_model'], DEFAULTS['threshold'], w)))
    ranked = rank(cutoffs, None, limit=20, explain_rows=50)
    return {'charts': [{'title': t, 'figure': _figure(fig), 'text': text} for t, fig, text in charts],
            'csv': {'name': ranked['name'], 'text': ranked['csv']}}


def _x(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _runs(text, bold=False, color=None, size=None):
    """Word runs from a line of text with **bold** markers (other Markdown marks are dropped)."""
    out = []
    for part in re.split(r'(\*\*[^*]+\*\*)', str(text)):
        if not part:
            continue
        strong = bold or (part.startswith('**') and part.endswith('**'))
        part = re.sub(r'[*`]', '', part)
        props = ('<w:b/>' if strong else '') + (f'<w:color w:val="{color}"/>' if color else '') + \
                (f'<w:sz w:val="{size}"/>' if size else '')
        out.append(f'<w:r>{"<w:rPr>" + props + "</w:rPr>" if props else ""}<w:t xml:space="preserve">{_x(part)}</w:t></w:r>')
    return ''.join(out)


def _para(text, style=None, **kw):
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
    return f'<w:p>{props}{_runs(text, **kw)}</w:p>'


def _md_paras(text):
    """Paragraphs and bullets from the short Markdown analysis texts."""
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(('- ', '* ')):
            out.append(_para('•  ' + line[2:], 'Bullet'))
        else:
            out.append(_para(re.sub(r'^#+\s*', '', line)))
    return ''.join(out)


def _docx_table(columns, rows, widths=None):
    widths = widths or [9638 // max(1, len(columns))] * len(columns)
    borders = ''.join(f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="D9DEE8"/>'
                      for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV'])

    def cell(value, width, header=False):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="EEF0FF"/>' if header else ''
        return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr><w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr>'
                f'{_runs(value, bold=header, color="1E1B4B" if header else None)}</w:p></w:tc>')

    head = '<w:tr><w:trPr><w:tblHeader/></w:trPr>' + ''.join(cell(c, wd, True) for c, wd in zip(columns, widths)) + '</w:tr>'
    body = ''.join('<w:tr><w:trPr><w:cantSplit/></w:trPr>' + ''.join(cell(v, wd) for v, wd in zip(r, widths)) + '</w:tr>' for r in rows)
    return (f'<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders>{borders}</w:tblBorders>'
            '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="90" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
            '<w:right w:w="90" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
            + ''.join(f'<w:gridCol w:w="{wd}"/>' for wd in widths) + f'</w:tblGrid>{head}{body}</w:tbl>' + _para(''))


def _docx_image(rid, number, width_px, height_px):
    cx = 5760720                          # 6.3 inches wide, in EMU
    cy = int(cx * height_px / max(1, width_px))
    if cy > 3474720:                      # at most 3.8 inches tall, so a chart and its heading share a page
        cx, cy = int(cx * 3474720 / cy), 3474720
    return (f'<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{number}" name="Chart {number}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{number}" name="Chart {number}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


DOCX_STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="1E293B"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="60"/></w:pPr><w:rPr><w:b/><w:color w:val="312E81"/><w:sz w:val="52"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="40"/></w:pPr><w:rPr><w:color w:val="64748B"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="120"/><w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="C7D2FE"/></w:pBdr><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="4F46E5"/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="0F172A"/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Bullet"><w:name w:val="Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="60"/><w:ind w:left="360" w:hanging="240"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="19"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Note"><w:name w:val="Note"/><w:basedOn w:val="Normal"/><w:pPr><w:pBdr><w:left w:val="single" w:sz="24" w:space="8" w:color="F5B301"/></w:pBdr><w:shd w:val="clear" w:color="auto" w:fill="FFF8E6"/><w:ind w:left="200"/><w:spacing w:before="120" w:after="120"/></w:pPr><w:rPr><w:color w:val="78350F"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Small"><w:name w:val="Small"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="64748B"/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>'''


def results_docx(images=None, name='', org='', link='', cutoffs=None):
    """The results as a Word document: key numbers, top reasons, charts (PNG images made by the page) with their
    analysis, the people most at risk, suggested actions, model comparison and the fairness check."""
    import datetime
    import io
    import zipfile
    trained, ctx = _need()
    w = ctx['wording']
    p = words(w)
    k = (STATE.get('train_info') or {}).get('kpis') or {}
    tops = top_drivers(trained, 5)
    sc = SCENARIOS[ctx['scenario']]
    today = datetime.date.today().strftime('%d %B %Y')
    body = [_para('ChurnSense results', 'Title'),
            _para(f"{w['people'].capitalize()} who {w['outcome_past']} · {ctx['name']} · {today}", 'Subtitle')]
    if name or org:
        body.append(_para('Prepared by ' + ', '.join(x for x in [name, org] if x), 'Subtitle'))
    if ctx['synthetic']:
        body.append(_para('**Test data.** This analysis uses a synthetic sample spreadsheet, not real people. '
                          "Use your institution's own data (My own data) for real results.", 'Note'))
    body.append(_para('Key numbers', 'Heading1'))
    rows = [[f"{w['people'].capitalize()} analysed", f"{k.get('rows', len(trained['prepared']['y'])):,}"],
            [f"Share who {w['outcome_past']}", k.get('rate', '')],
            ['Best model', f"{trained['best_model']} (ROC-AUC {k.get('auc', '')})"],
            [f"{'High risk' if p['leave'] else 'Highly likely'} in the test set", f"{k.get('high', '')} of {k.get('test', '')}"],
            ['Top reason', f"{k.get('top_reason', '')}: {k.get('top_phrase', '')}"]]
    body.append(_docx_table(['Measure', 'Value'], rows, [3600, 6038]))
    body.append(_para(p['why'], 'Heading1'))
    body += [_para(f'{i}.  **{c}**: {direction_phrase(trained, c, w)}', 'Bullet') for i, c in enumerate(tops, 1)]
    ranked = rank(cutoffs, None, limit=20, explain_rows=50)
    table = ranked['table']
    body.append(_para(f"{p['list'].capitalize()}: the top 20", 'Heading1'))
    cols = table['columns'][:5]

    def show(v):
        return f'{v:.0%}' if isinstance(v, float) and 0 <= v <= 1 else str(v)
    body.append(_docx_table(cols, [[show(v) for v in r[:5]] for r in table['rows'][:20]], [800, 1700, 1900, 1100, 4138]))
    body.append(_para(f"The full list of {ranked['total']:,} {w['people']} ({ranked['source']}) is in the CSV file "
                      f"**{ranked['name']}**.", 'Small'))
    actions = suggestions_for(tops, None, sc['rules'], w)
    if actions:
        body.append(_para(f"Suggested {p['actions']} actions to discuss", 'Heading1'))
        body += [_para('•  ' + a, 'Bullet') for a in actions[:8]]
    rels = []
    for i, img in enumerate(images or [], 1):
        rid = f'rIdImg{i}'
        rels.append((rid, f'media/chart{i}.png', _base64.b64decode(img['png'])))
        if i == 1:
            body.append(_para('Charts', 'Heading1'))
        body.append(_para(img.get('title', f'Chart {i}'), 'Heading2'))
        body.append(_docx_image(rid, i, img.get('width', 1000), img.get('height', 500)))
        body.append(_md_paras(img.get('text', '')))
    comp = comparison_table(trained, DEFAULTS['threshold']).round(3)
    body.append(_para('Model comparison', 'Heading1'))
    body.append(_docx_table(list(comp.columns), [[str(v) for v in r] for r in comp.values.tolist()]))
    try:
        _, _, fair = fairness_check(trained, DEFAULTS['threshold'], w)
        body.append(_para('Fairness check', 'Heading1'))
        body.append(_md_paras(fair))
    except Exception:
        pass
    body.append(_para('About these results', 'Heading1'))
    body.append(_para('These are patterns in past data, not certainties. Use them to start a conversation, never as the '
                      'only basis for a decision about a person.'))
    body.append(_para(re.sub(r'\*\*', '', METHOD_NOTE), 'Small'))
    body.append(_para('Made with ChurnSense' + (f': {link}' if link else '') + '. The analysis ran in the browser; '
                      'the spreadsheet was not uploaded anywhere.', 'Small'))
    ns = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
          'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
          'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
          'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<w:document {ns}><w:body>{"".join(body)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/>'
                '</w:sectPr></w:body></w:document>')
    rel_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
               + ''.join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                         f'Target="{target}"/>' for rid, target, _ in rels) + '</Relationships>')
    title = f"ChurnSense results: {w['people']} who {w['outcome_past']}"
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{_x(title)}</dc:title><dc:creator>{_x(name or "ChurnSense")}</dc:creator></cp:coreProperties>')
    types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
             '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
             '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
             '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 '</Relationships>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', types)
        z.writestr('_rels/.rels', root_rels)
        z.writestr('docProps/core.xml', core)
        z.writestr('word/document.xml', document)
        z.writestr('word/styles.xml', DOCX_STYLES)
        z.writestr('word/_rels/document.xml.rels', rel_xml)
        for _, target, data in rels:
            z.writestr(f'word/{target}', data)
    file_name = safe_name(f"ChurnSense_results_{ctx['scenario']}_{w['people']}") + '.docx'
    return {'name': file_name, 'b64': _base64.b64encode(buffer.getvalue()).decode('ascii')}


# ---------- the prompt box ----------
SCENARIO_WORDS = [
    ('Employees', r'\b(employees?|staff|workers?|attrition|workforce)\b'),
    ('Customers', r'\b(customers?|clients?|churn\w*|subscribers?)\b'),
    ('Students', r'\b(students?|pupils?|drop ?outs?|dropping out|dropped out)\b'),
    ('Teachers', r'\b(teachers?|educators?|faculty)\b'),
    ('Custom', r'\b(applicants?|enrol\w*|enroll\w*|admissions?|leads?)\b'),
]
JOIN_WORDS = r'\b(join\w*|enrol\w*|enroll\w*|sign(ed|s|ing)? ?up|convert\w*|subscrib\w*|buy\w*|bought|purchas\w*|accept\w*)\b'
LEAVE_WORDS = r'\b(leav\w*|left|churn\w*|quit\w*|drop\w* ?out|cancel\w*|resign\w*|attrition|exit\w*)\b'
MODEL_WORDS = {'Logistic Regression': r'\b(logistic( regression)?|lr)\b', 'Decision Tree': r'\b(decision trees?|single tree)\b',
               'K-Nearest Neighbors': r'\b(knn|k-?nn|nearest neighbou?rs?|k-nearest)\b',
               'Random Forest': r'\b(random forests?|forests?|rf)\b', 'Neural Network': r'\b(neural (net|network)s?|mlp|nn)\b'}
INTENTS = [
    ('all', r'\b(all|every)\b.*\b(graphs?|charts?|plots?|visuali[sz]ations?)\b'),
    ('help', r'^\s*(help|\?|what can (you|i) (do|ask))\b'),
    ('person', r'\b(explain|show|why (is|did|does|will)|predict for|look up|lookup)\b.*\b(person|employee|customer|student|teacher|applicant|member|id|row)\b\s*#?\s*[\w-]+'),
    ('whatif', r'\bwhat[ -]if\b|\bif (we|i|they)\b.*\b(change|set|were|was|had)\b'),
    ('who', r'\b(who|which|list|rank\w*|most likely|at[ -]risk|highest risk|top \d+|top (ten|twenty|five))\b'),
    ('why', r'\b(why|reasons?|drivers?|causes?|factors?|influenc\w*|what makes|predictors?)\b'),
    ('compare', r'\b(compare|comparison|best model|which model|accura\w*|metrics?|performance|roc|auc|f1|recall|precision)\b'),
    ('segments', r'\b(segments?|clusters?|groups?|kmeans|k-means)\b'),
    ('fairness', r'\b(fair\w*|bias\w*|discriminat\w*|sensitive|equal\w*)\b'),
    ('rate', r'\b(rate|percentage|share|how many)\b.*\bby\b'),
    ('explore', r'\b(histogram|distribution|box ?plot|scatter|bar chart|heatmap|correlation|plot|chart|graph)\b'),
    ('regression', r'\b(linear )?regression\b'),
    ('report', r'\b(report|download|export)\b'),
    ('score', r'\b(score|rank|predict)\b.*\.(csv|xlsx)\b'),
]
NUMBER_WORDS = {'five': 5, 'ten': 10, 'twenty': 20, 'fifty': 50}


def _mentioned_columns(text, columns):
    """Columns named in the text (exact name, or the name with spaces instead of CamelCase / underscores)."""
    found = []
    lowered = ' ' + re.sub(r'[^a-z0-9%]+', ' ', text.lower()) + ' '
    for c in sorted(columns, key=len, reverse=True):
        variants = {c.lower(), re.sub(r'[_\-]+', ' ', c).lower(),
                    re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', c).lower().replace('_', ' ')}
        for v in variants:
            v = re.sub(r'[^a-z0-9%]+', ' ', v).strip()
            if v and f' {v} ' in lowered:
                found.append((lowered.index(f' {v} '), c))
                lowered = lowered.replace(f' {v} ', ' ' * (len(v) + 2), 1)
                break
    return [c for _, c in sorted(found)]


def interpret_prompt(prompt, files=None):
    """Read a plain-English request: what to do (intents) and which settings it asks for."""
    text = prompt.strip()
    low = text.lower()
    found = {'intents': [i for i, pattern in INTENTS if re.search(pattern, low)], 'settings': {}}
    for name in (files or list(STATE['files'])):
        stem = os.path.splitext(name)[0].lower()
        if name.lower() in low or (len(stem) > 3 and re.search(rf'\b{re.escape(stem)}\b', low)):
            found.setdefault('files', []).append(name)
    for scenario, pattern in SCENARIO_WORDS:
        if re.search(pattern, low):
            found['scenario'] = scenario
            break
    joins, leaves = len(re.findall(JOIN_WORDS, low)), len(re.findall(LEAVE_WORDS, low))
    if joins or leaves:
        found['mode'] = 'join' if joins > leaves else 'leave'
    s = found['settings']
    if re.search(r'\b(thorough|accurate|wide|careful|best possible|slow)\b', low):
        s['preset'] = 'Thorough'
    elif re.search(r'\b(quick|fast|quickly)\b', low):
        s['preset'] = 'Quick'
    models = [m for m, pattern in MODEL_WORDS.items() if re.search(pattern, low)]
    if models and not re.search(r'\b(compare|which model|best model)\b', low):
        s['models'] = models
    if re.search(r'\ball (the )?models\b', low):
        s['models'] = list(MODEL_NAMES)
    m = re.search(r'threshold\s*(of|=|to|at)?\s*(0?\.\d+|\d+(\.\d+)?\s*%)', low)
    if m:
        value = m.group(2).replace(' ', '')
        s['threshold'] = float(value[:-1]) / 100 if value.endswith('%') else float(value)
    m = re.search(r'(\d+)\s*%\s*(for )?test|test( size| set)?\s*(of|=)?\s*(\d+)\s*%', low)
    if m:
        s['test_size'] = int(m.group(1) or m.group(5)) / 100
    m = re.search(r'\btop\s+(\d+|five|ten|twenty|fifty)\b|\b(\d+)\s+(most|highest|people|employees|customers|students|teachers|applicants)\b', low)
    if m:
        value = m.group(1) or m.group(2)
        found['top'] = NUMBER_WORDS.get(value, None) or int(value)
    return found


def _mentioned_target(text, df):
    """A target column asked for explicitly: "target is X", "outcome: X" or "predict X"."""
    columns = [str(c) for c in df.columns]
    for pattern in [r'\b(target|outcome|label)\s*(column)?\s*(is|=|:)\s*["\']?([^"\'?,.;!]+)',
                    r'\bpredict(ing|s)?\s+(the\s+)?(column\s+)?["\']?([^"\'?,.;!]+)']:
        m = re.search(pattern, text, re.I)
        if m:
            named = _mentioned_columns(m.group(4), columns)
            if named:
                return named[0]
    return None


def _person_key(text, ctx):
    if ctx is None:
        return None
    df = ctx['df']
    tokens = re.findall(r'[\w-]+', text)
    if ctx['id_column']:
        ids = set(df[ctx['id_column']].astype(str))
        for t in tokens:
            if t in ids:
                return t
    m = re.search(r'\brow\s*#?\s*(\d+)\b', text, re.I)
    return m.group(1) if m else None


def _changes(text, columns):
    """'what if OverTime = No and MonthlyIncome = 5000' -> {'OverTime': 'No', 'MonthlyIncome': '5000'}."""
    changes = {}
    for m in re.finditer(r'([A-Za-z_][\w ]*?)\s*(=|to|is|were|was)\s*["\']?([\w.\-]+)', text):
        named = _mentioned_columns(m.group(1), columns)
        if named:
            changes[named[-1]] = m.group(3)
    return changes


def _apply_value(value, current):
    if isinstance(current, (int, float, np.integer, np.floating)) and not isinstance(current, bool):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def run_prompt(prompt, options=None, ai=False):
    """Answer a request typed in the prompt box: choose the data and settings it asks for, train when needed, and
    return an answer with the matching charts and tables. Returns the options it used, for the page to show."""
    prompt = (prompt or '').strip()
    o = dict(options or {})
    found = interpret_prompt(prompt)
    intents = found['intents']
    did, blocks, downloads = [], [], []
    if 'help' in intents or not prompt:
        return {'html': md(HELP_TEXT), 'blocks': [], 'options': None, 'understood': True}

    # 1. which data
    files = found.get('files', [])
    score_file = None
    if len(files) >= 2 or ('score' in intents and files):
        score_file = files[-1]
        files = files[:-1] or ([o['file']] if o.get('file') else [])
    data_file = files[0] if files else o.get('file')
    if data_file is None and STATE['files']:
        data_file = next(iter(STATE['files']))
    if data_file is None:
        return {'html': md('**Add a spreadsheet first**: drag it into the box above, or click a sample button. '
                           'Then ask again.'), 'blocks': [], 'options': None, 'understood': True}
    changed_file = data_file != o.get('file') or STATE['ctx'] is None or STATE['ctx']['name'] != data_file
    if changed_file:
        setup = select_data(data_file, found.get('scenario') if found.get('scenario') and _guess_scenario(
            data_file, STATE['files'][data_file]['df'])[0] == 'Custom' else None, found.get('mode'))
        keep = {k: o[k] for k in ['preset', 'test_size', 'seed', 'scoring', 'models', 'balanced', 'logistic_max_iter',
                                 'mlp_max_iter', 'search', 'threshold', 'metric', 'cutoffs', 'k', 'kmax', 'missing',
                                 'exclude'] if k in o}
        o = {**setup_options(setup), **keep}
        did.append(f'Used **{data_file}** ({setup["scenario"]} scenario).')
    df = STATE['files'][data_file]['df']
    if found.get('mode') and o.get('scenario') == 'Custom' and found['mode'] != o.get('mode'):
        o['mode'] = found['mode']
        wording, rules = _custom_defaults(data_file, found['mode'])
        o['wording'], o['rules'] = wording, _table(rules)
        did.append(f"Switched to **{found['mode']}** mode.")
    target = _mentioned_target(prompt, df)
    if target and target != o.get('target'):
        o['target'] = target
        info = target_changed(target, o.get('drop'))
        o['positive'], o['features'] = info['positive'], info['features']
        did.append(f"Target column: **{target}** (outcome value '{info['positive']}').")
    for key, value in found['settings'].items():
        if key != 'threshold' and o.get(key) != value:
            o[key] = value
            if key == 'preset':
                o['search'] = _search_values(value)
            did.append(f"Set {key.replace('_', ' ')} to **{', '.join(value) if isinstance(value, list) else value}**.")
    if 'threshold' in found['settings']:
        o['threshold'] = found['settings']['threshold']
        did.append(f"Decision threshold: **{o['threshold']:.2f}**.")
    threshold = float(o.get('threshold', DEFAULTS['threshold']))

    # 2. train when needed
    retrained = False
    if STATE['trained'] is None or STATE['trained_key'] != _training_key(o):
        info = train(o)
        retrained = True
        did.append(f"Trained {len(info['models'])} models with the **{o.get('preset', DEFAULTS['preset'])}** search; "
                   f"best: **{info['best']}**.")
    trained, ctx = STATE['trained'], STATE['ctx']
    w = ctx['wording']
    p = words(w)
    answer = []

    # 3. answer
    handled = False
    key = _person_key(prompt, ctx) if ('person' in intents or 'whatif' in intents) else None
    if key is not None:
        handled = True
        row = _find_person(key)[trained['prepared']['features']]
        result = explain(_table(row), o.get('cutoffs'), o.get('rules'))
        answer.append(f"**{w['person'].capitalize()} {key}:** {p['score']} **{pct(result['probability'])}** "
                      f"({result['level']}).")
        blocks.append({'title': f"{w['person'].capitalize()} {key}", 'figure': result['figure'], 'html': result['html'],
                       'table': result['reasons']})
        changes = _changes(prompt.split('if', 1)[-1], list(row.columns)) if 'whatif' in intents else {}
        if changes:
            new = row.copy()
            for col, value in changes.items():
                new[col] = _apply_value(value, row.iloc[0][col])
            _, text = what_if(trained, STATE['person'], new, w, _cutoffs(o.get('cutoffs')))
            answer.append('**What-if:** ' + text)
    if 'who' in intents or 'score' in intents:
        handled = True
        top = found.get('top', 10)
        ranked = rank(o.get('cutoffs'), score_file, limit=top, explain_rows=max(top, 50))
        answer.append(f"**Top {min(top, ranked['total'])} by {p['score']}** in {ranked['source']}"
                      + (' (the model did not see these people while training)' if not score_file else '') + '.')
        blocks.append({'title': f"{p['list'].capitalize()} ({ranked['source']})", 'figure': None,
                       'html': ranked['notes_html'], 'table': ranked['table']})
        downloads.append({'name': ranked['name'], 'text': ranked['csv'], 'label': f"Download the full {p['list']} (CSV)"})
    if 'why' in intents:
        handled = True
        reasons = '; '.join(f"**{c}** ({direction_phrase(trained, c, w)})" for c in top_drivers(trained, 5))
        answer.append(f"**{p['why']}** (best model {trained['best_model']}): {reasons}.")
        blocks += why_view()['blocks'][:3]
    if 'compare' in intents:
        handled = True
        view = compare_view(o.get('metric', 'ROC-AUC'), threshold)
        r = trained['results'][trained['best_model']]
        answer.append(f"**Best model: {trained['best_model']}** (CV {trained['settings']['scoring']} "
                      f"{r['cv_score']:.3f}). All models are compared below.")
        blocks.append({'title': 'All models', 'figure': None, 'html': '', 'table': view['table']})
        blocks += view['blocks'][:1]
    if 'segments' in intents:
        handled = True
        view = segments_view(None, o.get('k', 3), o.get('kmax', 8), None, None, threshold, o.get('rules'))
        answer.append('**Segments** (groups of similar people) are below.')
        blocks += view['blocks'][1:]
        blocks.append({'title': 'Segments', 'figure': None, 'html': '', 'table': view['table']})
    if 'fairness' in intents:
        handled = True
        view = fairness_view(threshold)
        answer.append('**Fairness check** across the sensitive columns is below.')
        blocks.append({**view['block'], 'table': view['table']})
    columns = [str(c) for c in ctx['df'].columns]
    named = [c for c in _mentioned_columns(prompt, columns) if c != trained['prepared']['target']]
    if 'rate' in intents and named:
        handled = True
        blocks.append(outcome_rate(named[0]))
        answer.append(f"**{p['outcome_rate'].capitalize()} by {named[0]}** is below.")
    elif 'explore' in intents and named:
        handled = True
        kind = ('Correlation heatmap' if re.search(r'correlation|heatmap', prompt, re.I) else
                'Scatter' if re.search(r'scatter|\bvs\.?\b|against', prompt, re.I) and len(named) > 1 else
                'Box' if re.search(r'box', prompt, re.I) and len(named) > 1 else
                'Bar' if re.search(r'\bbar\b', prompt, re.I) else 'Histogram')
        x = named[0]
        y = named[1] if kind in ('Scatter', 'Box') and len(named) > 1 else None
        if kind == 'Box' and y and is_numeric(ctx['df'][x]) and not is_numeric(ctx['df'][y]):
            x, y = y, x
        blocks.append(explore(kind, x, y, trained['prepared']['target']))
        answer.append(f'**{kind}** of {x}' + (f' and {y}' if y else '') + ' is below.')
    elif 'explore' in intents and re.search(r'correlation|heatmap', prompt, re.I):
        handled = True
        blocks.append(explore('Correlation heatmap'))
        answer.append('**Correlation heatmap** of the numeric columns is below.')
    if 'regression' in intents:
        numeric = [c for c in named if c in numeric_columns(ctx['df'])]
        if numeric:
            handled = True
            features = numeric[1:] or default_regression_columns(ctx['df'][numeric_columns(ctx['df'])], numeric[0])[1]
            result = regression(numeric[0], features, o.get('test_size', 0.2), o.get('seed', 42))
            blocks.append({**result['block'], 'table': result['coefficients']})
            answer.append(f'**Linear Regression of {numeric[0]}** is below.')
    open_tab = None
    if 'all' in intents:
        handled = True
        view = all_graphs(threshold, o.get('metric', 'ROC-AUC'), o.get('rules'))
        open_tab = 'all'
        answer.append(f"**All {view['count']} graphs** are in the **★ All graphs** tab below, grouped by model "
                      f"({view['original']} of them are the graphs from the original notebook).")
    if 'report' in intents:
        handled = True
        rep = report(threshold, o.get('metric', 'ROC-AUC'), o.get('rules'))
        downloads.append({'name': rep['name'], 'text': rep['html'], 'label': 'Download the summary report (HTML)',
                          'type': 'text/html'})
        answer.append('**Summary report** with every graph and its analysis is ready to download below.')
    if not handled:
        # a general request: the overview
        summary = overall_summary(trained, w, threshold, _rules(o.get('rules')))
        answer.append(summary)
        blocks.append(why_view()['blocks'][0])
        ranked = rank(o.get('cutoffs'), score_file, limit=10, explain_rows=50)
        blocks.append({'title': f"{p['list'].capitalize()} (top 10, {ranked['source']})", 'figure': None,
                       'html': '', 'table': ranked['table']})
        downloads.append({'name': ranked['name'], 'text': ranked['csv'], 'label': f"Download the full {p['list']} (CSV)"})
        if not retrained and not did and len(prompt.split()) > 2:
            answer.append('*I did not recognise a specific request, so here is the overview'
                          + (' (the AI answer below responds to your question).*' if ai else
                             '. Type **help** to see what I understand, or add a free Gemini key for free-form questions.*'))
    if ctx['synthetic']:
        answer.append(SYNTHETIC_NOTICE)
    answer.append(NOTICE)
    html = ''
    if did:
        html += '<div class="did"><b>What I did:</b>' + md('\n'.join(f'- {d}' for d in did)) + '</div>'
    html += md('\n\n'.join(answer))
    return {'html': html, 'blocks': blocks, 'downloads': downloads, 'options': o, 'retrained': retrained,
            'understood': handled, 'open_tab': open_tab}


HELP_TEXT = '''### What you can ask
- **Analyse a file:** "Which employees are most likely to leave? Target is Attrition" or "analyse customers_SYNTHETIC.csv"
- **Who:** "top 20 students most likely to drop out", "rank the people in new_hires.xlsx"
- **Why:** "why do customers churn?", "what are the main reasons?"
- **Models:** "compare the models", "which model is best?", "use only random forest", "use thorough"
- **One person:** "explain employee 1043", "explain row 12", "what if OverTime = No for employee 1043"
- **Groups and fairness:** "show the segments", "is the model fair across gender?"
- **Charts:** "histogram of MonthlyIncome", "scatter tenure vs MonthlyCharges", "churn rate by Contract", "correlation heatmap"
- **Every graph:** "show all graphs" (also the **★ All graphs** tab)
- **Other:** "threshold 0.4", "25% test", "regression of MonthlyIncome", "download the report"

With a free Gemini key (in **AI answers**), any other question is answered by AI from a summary of the results.'''


def setup_options(setup):
    """Default options for a newly selected file (from select_data)."""
    return {'file': setup['file'], 'scenario': setup['scenario'], 'mode': setup['mode'], 'wording': setup['wording'],
            'target': setup['target'], 'positive': setup['positive'], 'drop': setup['drop'],
            'features': setup['features'], 'sensitive': setup['sensitive'], 'exclude': DEFAULTS['exclude_sensitive'],
            'missing': DEFAULTS['missing_strategy'], 'rules': setup['rules'], 'preset': BROWSER_PRESET,
            'test_size': DEFAULTS['test_size'], 'seed': DEFAULTS['random_state'], 'scoring': DEFAULTS['scoring'],
            'models': list(MODEL_NAMES), 'balanced': DEFAULTS['balanced'],
            'logistic_max_iter': DEFAULTS['logistic_max_iter'], 'mlp_max_iter': DEFAULTS['mlp_max_iter'],
            'search': _search_values(BROWSER_PRESET), 'threshold': DEFAULTS['threshold'], 'metric': 'ROC-AUC',
            'cutoffs': list(DEFAULTS['level_cutoffs']), 'k': DEFAULTS['clustering']['n_clusters'],
            'kmax': DEFAULTS['clustering']['max_k']}


BROWSER_PRESET = 'Quick'   # the browser uses one CPU core: Quick takes under a minute, Thorough several minutes


def defaults():
    return {'models': MODEL_NAMES, 'presets': list(SEARCH_PRESETS), 'preset': BROWSER_PRESET,
            'scenarios': {k: v['name'] for k, v in SCENARIOS.items()}, 'samples': {k: v['sample'] for k, v in SCENARIOS.items()},
            'scoring': list(SCORING), 'missing': MISSING_STRATEGIES, 'conditions': CONDITIONS,
            'search': _search_values(BROWSER_PRESET), 'defaults': {k: DEFAULTS[k] for k in
            ['test_size', 'random_state', 'scoring', 'balanced', 'logistic_max_iter', 'mlp_max_iter', 'threshold',
             'level_cutoffs', 'missing_strategy', 'exclude_sensitive']},
            'clustering': DEFAULTS['clustering'], 'help_html': md(HELP_TEXT), 'notice_html': md(NOTICE)}


def md_html(text):
    return {'html': md(text)}


API = {f.__name__: f for f in [add_file, remove_file, select_data, target_changed, custom_mode, preset_values, train,
                               model_view, compare_view, why_view, segments_view, cluster_axes, fairness_view,
                               explore, outcome_rate, explain, whatif, load_person, rank, results_csv, report,
                               chart_list, make_zip, regression, ai_context, run_prompt, defaults, md_html,
                               setup_info, trained_info, all_graphs, email_summary, share_pack, results_docx]}


def api_call(name, args_json='{}'):
    """Entry point for the page: run API[name](**args) and return JSON {ok, result} or {ok: false, error}."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*JsProxy.*')
            result = API[name](**_json.loads(args_json or '{}'))
        return _json.dumps({'ok': True, 'result': _clean(result)})
    except (ValueError, KeyError, FileNotFoundError) as error:
        return _json.dumps({'ok': False, 'error': str(error).strip("'\"")})
    except Exception as error:
        return _json.dumps({'ok': False, 'error': f'Something went wrong ({type(error).__name__}): {error}'})

