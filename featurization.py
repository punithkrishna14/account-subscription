import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn.manifold import TSNE
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

#Create a new col for 'balance' feature which indicates negative or positive balance
def balance_status(row):
    value = row['balance']
    if value < 0:
        return 'neg'
    else :
        return 'pos'

#Convert categorical string features to numeric representations    
def feature_categorization(feature,data):
    feat = data[feature].unique()
    if feature == 'is_success':
        feat_dict = {feat[i]: i for i in range(0,len(feat))}
    else:
        feat_dict = {feat[i]: i+1 for i in range(0,len(feat))}
    data[feature+'_cat'] = data[feature].apply(lambda row: feat_dict[row])
    data.drop(feature, axis=1, inplace=True)

#Convert all string feature values to numeric
def featurization_dataset(data):
    features = ['job','marital','education','default','housing','loan','contact','month','poutcome','is_success','balance_status']
    
    data['balance_status'] = data.apply(balance_status, axis = 1)
    print data.columns
    data.drop('day', axis=1, inplace=True)
    
    for i in features:
        feature_categorization(i,data)


#Categorial to one hot encoding
def cat_to_ohe(feature,data):

    enc = OneHotEncoder()
    feature_reshape = data[feature].values.reshape(-1, 1)
    cat_feat = enc.fit_transform(feature_reshape).toarray()
    
    #-1 to avoid dummy variable trap
    featr = [feature+' '+str(i+1) for i in range(0,cat_feat.shape[1]-1)]
    for i in range(0,len(featr)):
        data[featr[i]] = cat_feat[:,i]

#Convert 'pdays' feature to binary feature and one hot encode all categorical features
def category_ohe(data):
    features = ['job','marital','education','default','housing','loan','contact','month','poutcome','is_success','balance_status']
    
    prev = data['pdays'].values
    prev = [1 if i == -1 else 0 for i in prev]
    data['pdays'] = prev
    
    features.remove('is_success')
    features_cat = [i+'_cat' for i in features]
        
    for i in features_cat:
        cat_to_ohe(i,data)
        data.drop(i, axis=1, inplace=True)


#Standardization of continuous variables:
#Train and test are standardized separately
def scale_feature(feature,df):
    values = df[feature].values
    values = values.reshape((len(values), 1))
     # train the standardization
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    standardized = scaler.transform(values)
    
    return np.array(standardized)

#Standardize and return 75:25 split dataset ready for modelling
def standardization_train_test_split(data):
    features = ['age', 'balance','duration','campaign', 'previous']
    
    x_train, x_test, y_train, y_test = train_test_split(data, data['is_success_cat'], test_size=0.25, random_state=30)
    x_train.drop('is_success_cat', axis=1, inplace=True)
    x_test.drop('is_success_cat', axis=1, inplace=True)
    
    for i in features:
        x_train[i+'_scaled'] = scale_feature(i,x_train)
        x_test[i+'_scaled'] = scale_feature(i,x_test)
        
        x_train.drop(i, axis=1, inplace=True)
        x_test.drop(i, axis=1, inplace=True)
    
    
    
    new_features = list(x_train.columns)
    print "New features: \n{} \n\n Number of features : {}".format(new_features,len(new_features))
    
    return [x_train,x_test,y_train,y_test]
    

#tSNE is optional - takes time to generate
def tsne_embeddings(data):

    n_sne = 7000

    time_start = time.time()

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x_train.values)

    print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
    
    df_tsne = pd.DataFrame()
    x_tsne = tsne_results[:,0]
    y_tsne = tsne_results[:,1]
    label = y_train.values
    colors = ["yellow", "green"]
    groups = ("Term deposit bought","Term deposit not bought") 
    # Plot
    data = (x_tsne,y_tsne)
    colors = [colors[i] for i in label]
    
    plt.scatter(x_tsne, y_tsne , s=5, c=colors, alpha=0.9)

    plt.title('tSNE embeddings differentiated by term deposit')
    plt.legend(loc=2)
    plt.savefig(i+'_boxplot'+'.png')
    plt.show()
 
#Confusion matrix for evaluating and scrutinizing models
def show_confusion_matrix(C,classifier,class_labels=['0','1']):
    #Draws confusion matrix with associated metrics.
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tp = C[0,0]; fp = C[0,1]; fn = C[1,0]; tn = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.savefig(classifier+'_Confusion_mat')
    plt.show()

#Modelling stage which prints precison , recall, ROC curve 
def model(x_train,y_train,x_test,y_test,clf,classifier):

    clf.fit(list(x_train.values),list(y_train.values))
    y_pred = clf.predict(list(x_test.values))
    conf_mat = confusion_matrix(list(y_test), list(y_pred))

    print "Confusion matrix:\n",conf_mat

    #print confusion matrix
    show_confusion_matrix(conf_mat,classifier,['Negative','Positive'])

    #Precision and recall
    tp = conf_mat[0,0]; fp = conf_mat[0,1]; fn = conf_mat[1,0]; tn = conf_mat[1,1];

    precision = 100*float(tp)/(tp+fp)
    recall = 100*float(tp)/(tp+fn)

    print "Precision :",precision
    print "Recall :",recall
    
    try:
        # Compute ROC curve and ROC area for each class
        probs = clf.predict_proba(list(x_test.values))
        preds = probs[:,1]
        fpr, tpr, threshold = roc_curve(list(y_test), preds)
        roc_auc = auc(fpr, tpr)

        #Plot ROC
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(classifier+'_ROC')
        plt.show()

        
    except:
        print "ROC Unavailable"
