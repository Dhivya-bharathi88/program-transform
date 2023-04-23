# Ex.No-6-Feature-Transformation
# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
# STEP-1:
Read the given Data.

# STEP-2:
Clean the Data Set using Data Cleaning Process.

# STEP-3:
Apply Feature Transformation techniques to all the feature of the data set.

# STEP-4:
Save the data to the file.

# CODE:
# FUNCTION TRANSFORMATION:
# LOG TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df['ModeratePositiveSkew']=np.log(df.ModeratePositiveSkew)

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')

plt.show()

# RECIPROCAL TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

df['HighlyPositiveSkew']=1/df.HighlyPositiveSkew

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df['HighlyNegativeSkew']=1/df.HighlyNegativeSkew sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45') plt.show()

df['ModeratePositiveSkew']=1/df.ModeratePositiveSkew sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45') plt.show()

df['ModerateNegativeSkew']=1/df.ModerateNegativeSkew sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45') plt.show()

# SQUARE ROOT TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

df['HighlyPositiveSkew']=np.sqrt(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df['ModeratePositiveSkew']=np.sqrt(df.ModeratePositiveSkew) sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45') plt.show()

# POWER TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import PowerTransformer

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

transformer=PowerTransformer("yeo-johnson")

df['HighlyPositiveSkew']=pd.DataFrame(transformer.fit_transform(df[['HighlyPositiveSkew']]))

sm.qqplot(df['HighlyPositiveSkew'],line='45')

plt.show()

df['HighlyNegativeSkew']=pd.DataFrame(transformer.fit_transform(df[['HighlyNegativeSkew']]))

sm.qqplot(df['HighlyNegativeSkew'],line='45')

plt.show()

df['ModeratePositiveSkew']=pd.DataFrame(transformer.fit_transform(df[['ModeratePositiveSkew']]))

sm.qqplot(df['ModeratePositiveSkew'],line='45')

plt.show()

df['ModerateNegativeSkew']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df['ModerateNegativeSkew'],line='45')

plt.show()

# QUANTILE TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

qt=QuantileTransformer(output_distribution='normal')

df['HighlyPositiveSkew']=pd.DataFrame(qt.fit_transform(df[['HighlyPositiveSkew']]))

sm.qqplot(df['HighlyPositiveSkew'],line='45')

plt.show()

df['HighlyNegativeSkew']=pd.DataFrame(qt.fit_transform(df[['HighlyNegativeSkew']]))

sm.qqplot(df['HighlyNegativeSkew'],line='45')

plt.show()

df['ModeratePositiveSkew']=pd.DataFrame(qt.fit_transform(df[['ModeratePositiveSkew']]))

sm.qqplot(df['ModeratePositiveSkew'],line='45')

plt.show()

df['ModerateNegativeSkew']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df['ModerateNegativeSkew'],line='45')

plt.show()

# OUTPUT:
# FUNCTION TRANSFORMATION:
# LOG TRANSFORMATION:
![Screenshot (151)](https://user-images.githubusercontent.com/128019999/233839180-ecf4aa31-767a-459c-87ef-7746b5c9f4b7.png)
![Screenshot (152)](https://user-images.githubusercontent.com/128019999/233839192-4b9b3c94-5689-40c8-a71c-1fca6c896560.png)
# RECIPROCAL TRANSFORMATION:
![Screenshot (152)](https://user-images.githubusercontent.com/128019999/233839912-53690149-52b0-4935-854f-073cde07004d.png)
![Screenshot (153)](https://user-images.githubusercontent.com/128019999/233839209-3ae57da3-e752-4d46-84ee-5c202e1ad75a.png)
![Screenshot (154)](https://user-images.githubusercontent.com/128019999/233839229-47464d79-9160-45ad-bb6f-a9bfa876111c.png)
![Screenshot (155)](https://user-images.githubusercontent.com/128019999/233839243-6763517d-c11a-4192-87c5-81a8d16b5940.png)
# SQUARE ROOT TRANSFORMATION:
![Screenshot (156)](https://user-images.githubusercontent.com/128019999/233839253-6dfd1c6a-7814-4d92-ba1f-e27d237bd6e2.png)
![Screenshot (157)](https://user-images.githubusercontent.com/128019999/233839262-2ce064d1-879e-46b2-bc75-e2af05b8a887.png)
# POWER TRANSFORMATION:
![Screenshot (158)](https://user-images.githubusercontent.com/128019999/233839275-a68952b9-4944-4dc5-a07b-8f5ead323143.png)
![Screenshot (159)](https://user-images.githubusercontent.com/128019999/233839286-ba2e4242-a10f-466d-9ca0-bab2564f8404.png)
![Screenshot (160)](https://user-images.githubusercontent.com/128019999/233839296-8a1a067f-f3a9-4894-bc2b-32cc97d11a18.png)
![Screenshot (161)](https://user-images.githubusercontent.com/128019999/233839303-a8600d93-1513-4498-af49-da43cf90fbd4.png)
# QUANTILE TRANSFORMATION:
![Screenshot (162)](https://user-images.githubusercontent.com/128019999/233839309-098b1df4-a10c-498d-816e-89bc40d53c1b.png)
![Screenshot (163)](https://user-images.githubusercontent.com/128019999/233839317-b57b3b33-66e4-4d6c-81a7-660cf047ba8e.png)
![Screenshot (164)](https://user-images.githubusercontent.com/128019999/233839341-45c97ba0-1f49-4369-975a-6b265be2a612.png)
![Screenshot (165)](https://user-images.githubusercontent.com/128019999/233839147-7ac8a290-4730-4ca3-a2db-b2af02e82cfa.png)
# RESULT:
Thus, the Feature Transformation for the given data set is executed and output was verified successfully.
