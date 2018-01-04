import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from matplotlib import pyplot
from pylab import rcParams
    

#generates bar plot for a given feature
def bar_pot_univariate(data,feature,verbose):
    width = 0.8
    colors = ['#CD5C5C','#FFA07A','#FFD700','#d62728','#F0E68C','#7FFF00','#90EE90','#AFEEEE','#5F9EA0','#DA70D6',\
              '#FF00FF','#F5F5DC','#D3D3D3','#FFE4C4','#C1A2A2']

    categories = data[feature].unique()
    
    #bar plot for term deposit subcribed and term deposit not subscribed
    prod_not_bought = [data[(data.is_success == 'no') & (data[feature] == i)].shape[0] for i in categories]
    prod_bought = [data[(data.is_success == 'yes') & (data[feature] == i)].shape[0] for i in categories]

    feature_numeric = [i for i in range(0,len(categories))]
    #prod_bought,prod_not_bought
	
	#choose color of plot randomly    
    plot_color = np.random.randint(0,len(colors))
    
    fig, ax = plt.subplots(1,1)
    p1 = plt.bar(feature_numeric, prod_not_bought, width, color=colors[plot_color])
    p2 = plt.bar(feature_numeric, prod_bought, width,
                 bottom=prod_not_bought)
    ax.set_xticks(feature_numeric)
    # Set ticks labels for x-axis
    ax.set_xticklabels(categories, rotation='vertical', fontsize=18)
    rcParams['figure.figsize'] = 9, 9

    plt.ylabel('Number of persons from each '+feature+' category')
    plt.xlabel(feature+' categories')
    plt.title('Comparsion among '+feature+' categories, people who have subscribed to term deposit vs people who haven\'t')
    plt.legend((p2[0], p1[0]), ('Subscribed to term deposit', \
                                'Haven\'t subscribed to term deposits'))
    
    print "Saving plot as png"
    plt.savefig(feature+'_category'+'.png')
    
    if verbose == 1:
        plt.show()
        pos = 0
        ratio = [round(i*100.0/j,2) for i,j in zip(prod_bought,prod_not_bought)]
    
    	#Have some stats along with the graphs describing the distribution of a particular feature among 
    	#people who've subscribed to term depsoit and people who haven't

        print "Descriptive Stats : Feature - {}\n".format(feature)
        for i,j,k,l in sorted(zip(ratio,categories,prod_bought,prod_not_bought)):
            pos+=1
            print "{}.{} Feature category term deposit subscription :No {} vs Yes {} | Ratio - {}".format(pos,j,l,k,i)


#generates bar graph for categorical features
def generate_bar_categorical(dataframe,verbose):
    features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome','contact', 'month']

    for i in features:
        bar_pot_univariate(dataframe,i,verbose)


#PDF plots of numeric and continuous variables
def kde_plot(data,feature,verbose):
    colors_light = ['#FFB4F7','#C5C3FA','#C3EBFA','#B0F9A5','#FBFF95','#FF9595','#FFD495']
    colors_dark = ['#FFFF0E','#FD1591','#1F50F2','#F90F0F','#FF4500','#008000','#FF8C00']

    successful = np.sort(data[data['is_success'] == 'yes'][feature].values)
    unsuccessful = np.sort(data[data['is_success'] == 'no'][feature].values)
    
    #choose max of 2000 bins
    bins = min([max([max(successful),max(unsuccessful)]),2000])
    
    print "Feature: {}".format(feature)
    pyplot.hist(data[data['is_success'] == 'yes'][feature][0:len(successful)-10], bins, alpha=0.5, label='Term subscribed',color=[colors_dark[np.random.randint(1,len(colors_dark))]])
    pyplot.hist(data[data['is_success'] == 'no'][feature][0:len(unsuccessful)-10], bins, alpha=0.5, label='Term not subscribed',color=[colors_light[np.random.randint(1,len(colors_light))]])
    pyplot.legend(loc='upper right')
    plt.title('Pdf plots of '+feature+' category for people having subscribed to term deposits and people who\
    haven\'t')
    
    print "Saving plot as png"
    plt.savefig(feature+'_category'+'.png')
    
    if verbose == 1:
        pyplot.show()
    
        suc_indices = [i for i in range(len(successful)/10,len(successful),len(successful)/10)]
        unsuc_indices = [i for i in range(len(unsuccessful)/10,len(unsuccessful),len(unsuccessful)/10)]
        
        #Describe Parametric stats of continuous variables
        print "Percentile values :"
        percentile = 0
        print "Feature {} : Parametric Stats".format(feature)
        for i,j in zip(suc_indices,unsuc_indices):
            percentile+=10
            print "{} percentile : successful {} | Unsuccesful {}".format(percentile,successful[i],unsuccessful[j])
        print "\n"

#Create PDF plots of continuous variables
def generate_distplots_continuous(dataframe,verbose):
    continuous_variables = ['age','duration','campaign']

    for i in continuous_variables:
        kde_plot(dataframe,i,verbose)


#Phone call duration effect on buying products:
def last_call_duration_kde(data,verbose):
    phone_success = data[data['is_success'] == 'yes']['duration'].values
    phone_unsuccess = data[data['is_success'] == 'no']['duration'].values

    phone_success = phone_success.reshape(1,len(phone_success))
    phone_unsuccess = phone_unsuccess.reshape(1,len(phone_unsuccess))
    phone_call_duration = np.append(phone_success,phone_unsuccess)
    df = pd.DataFrame(phone_call_duration,columns = ['phone_calls'])

    with sns.plotting_context("notebook",font_scale=1.5):
        sns.set_style("whitegrid")
        dims = (11.7, 8.27)
        fig, ax = pyplot.subplots(figsize=dims)
        sns.distplot(df["phone_calls"],
                    bins=2000,
                     kde=True,
                     color="turquoise")
        sns.plt.title("Duration from last call  - Distribution")
        plt.ylabel("Count")
        
        print "Saving plot as png file.."
        plt.savefig('Duration from last call Distribution'+'.png')
        if verbose == 1:
            plt.show()
    
    if verbose == 1:
        phone_success = phone_success[0]
        phone_unsuccess = phone_unsuccess[0]

        #Sucessful calls vs unsuccessful calls as the duration between calls increases
        durations = [i for i in range(100,1000,50)]
        calls = [[100.0*len([j for j in phone_success if j >= i])/len(phone_success),\
                  100.0*len([k for k in phone_unsuccess if k >= i])/len(phone_unsuccess)] \
                 for i in durations]

        for i,j in zip(calls,durations):
            print " 'Last call duration' greater than {} s : Successful - {}% Unsuccessful - {}%".\
            format(j,round(i[0],2),round(i[1],2))

#Box plots for continuous variables:
def boxplot_continuous(data,verbose):
    features_continuous = ['age','duration','campaign']
    for i in features_continuous:
        ax = sns.boxplot(x="is_success", y=i, data=data)
        plt.savefig(i+'_boxplot'+'.png')
        if verbose == 1:
            plt.title('Box plot of '+i)
            plt.show()

