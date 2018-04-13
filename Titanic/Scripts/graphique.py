import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# example data
x = df['AgeFill']

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, facecolor='green', alpha=0.5)
# add a 'best fit' line
plt.xlabel('Age')
plt.ylabel('Nombre')
plt.title(r'Age')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.10)
plt.show()


x = df

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, ncols=2 , figsize=(8, 4))

ax0.hist(df['AgeFill'], 30, histtype='bar', facecolor='g', alpha=0.75)
ax0.set_title('AgeFill')

ax1.hist(df['Pclass'], histtype='bar', bins=3,facecolor='g', alpha=0.75,rwidth=1)
ax1.set_title('Pclass')
ax1.set_xticks([1, 2, 3])
#ax1.set_xlim([1,3])
ax1.set_ylim([0, 700])
    
ax2.hist(df['Mother'], bins=2,histtype='bar', facecolor='r')
ax2.set_title('Mother')
ax2.set_ylim([0, 900])
ax2.set_xticks([0,1])

ax3.hist(df['FamilySize'], 10,histtype='bar', facecolor='y', alpha=0.75)
ax3.set_title('FamilySize')
ax3.set_ylim([0, 700])
ax3.set_xticks([1, 2, 3,4,5,6,7,8,9,10])

ax4.hist(df['Gender'], 20, histtype='stepfilled', facecolor='b', alpha=0.75)
ax4.set_title('Gender')
ax4.set_ylim([0, 700])

plt.tight_layout()
plt.show()

#df.groupby(df['Pclass']).aggregate(np.sum)
#table = pd.pivot_table(df, index=['Pclass'],
#               columns=['Survived'], aggfunc=np.sum)

import numpy as np
import matplotlib.pyplot as plt

N = 2
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='r', yerr=menStd)
p2 = plt.bar(ind, womenMeans, width, color='y',
             bottom=menMeans, yerr=womenStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()
