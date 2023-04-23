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
df

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
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

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
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

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
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

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
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

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
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

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
df['ModeratePositiveSkew']=np.log(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

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
df['HighlyPositiveSkew']=1/df.HighlyPositiveSkew
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

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
df['HighlyNegativeSkew']=1/df.HighlyNegativeSkew
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

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
df['ModeratePositiveSkew']=1/df.ModeratePositiveSkew
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

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
df['ModerateNegativeSkew']=1/df.ModerateNegativeSkew
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import QuantileTransformer
import scipy.stats as stats
from google.colab import files
uploaded = files.upload()
df=pd.read_csv("Data_to_Transform1.csv")
df['HighlyPositiveSkew']=np.sqrt(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from google.colab import files
uploaded = files.upload()
df=pd.read_csv("Data_to_Transform1.csv")
df['ModeratePositiveSkew']=np.sqrt(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

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
df['HighlyNegativeSkew']=pd.DataFrame(transformer.fit_transform(df[['HighlyNegativeSkew']]))
sm.qqplot(df['HighlyNegativeSkew'],line='45')
plt.show()

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
df['ModeratePositiveSkew']=pd.DataFrame(transformer.fit_transform(df[['ModeratePositiveSkew']]))
sm.qqplot(df['ModeratePositiveSkew'],line='45')
plt.show()

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
df['ModerateNegativeSkew']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df['ModerateNegativeSkew'],line='45')
plt.show()

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
df['HighlyNegativeSkew']=pd.DataFrame(qt.fit_transform(df[['HighlyNegativeSkew']]))
sm.qqplot(df['HighlyNegativeSkew'],line='45')
plt.show()

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
df['ModeratePositiveSkew']=pd.DataFrame(qt.fit_transform(df[['ModeratePositiveSkew']]))
sm.qqplot(df['ModeratePositiveSkew'],line='45')
plt.show()

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
df['ModerateNegativeSkew']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df['ModerateNegativeSkew'],line='45')
plt.show()
