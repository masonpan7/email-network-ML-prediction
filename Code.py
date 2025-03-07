G = pickle.load(open('assets/email_prediction_NEW.txt', 'rb'))

print(f"Graph with {len(nx.nodes(G))} nodes and {len(nx.edges(G))} edges")
def salary_predictions():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import math

    num_of_nodes = G.number_of_nodes()
  
    #Feature 1
    degree_centrality = nx.degree_centrality(G)
    degree_centrality = dict(sorted(degree_centrality.items()))
    list_of_degree_centrality = []
    for key,value in degree_centrality.items():
        list_of_degree_centrality.append(value)
      
    #Feature 2
    clustering_coefficients = []
    sorted_nodes = sorted(G.nodes())
    for nodes in G.nodes():
        clust_coeff = nx.clustering(G, nodes)
        clustering_coefficients.append(clust_coeff)
    
    #Feature 3
    page_rank = []
    page_rank_values = nx.pagerank(G)
    page_rank_values = dict(sorted(page_rank_values.items()))
    for key,value in page_rank_values.items():
        page_rank.append(value)
    
    #Feature 4
    department = []
    for node in sorted_nodes:
        department.append(G.nodes[node]["Department"])

    #Feature 5
    management_salary = []
    for node in sorted_nodes:
        management_salary.append(G.nodes[node]["ManagementSalary"])
        
    df = pd.DataFrame(index=sorted_nodes)
    df['degree_centrality'] = list_of_degree_centrality
    df['clustering_coefficients'] = clustering_coefficients
    df['page_rank'] = page_rank
    df['department'] = department
    df['management_salary'] = management_salary
    
    df_test = pd.DataFrame()
    for index, row in df.iterrows():
        if math.isnan(row['management_salary']):
            df_test = df_test.append(row)
            df = df.drop(index)
    
    scaler = StandardScaler()
    
    X_train = df[['degree_centrality', 'clustering_coefficients', 'page_rank', 'department']]
    y_train = df['management_salary']
    X_train_scaled = scaler.fit_transform(X_train)
    X_test = df_test[['degree_centrality', 'clustering_coefficients', 'page_rank', 'department']]
    X_test_scaled = scaler.fit_transform(X_test)
    
    clf = RandomForestClassifier(max_depth = 7, n_estimators = 75).fit(X_train,y_train)
    
    y_predicted = clf.predict_proba(X_test)[:,1]
    y_predicted_list = list(y_predicted)
    result = pd.Series(y_predicted_list, index = df_test.index)
    return result
    
    
salary_predictions()

future_connections = pd.read_csv('assets/Future_Connections.csv', index_col=0, converters={0: eval})
future_connections.head(10)

def new_connections_predictions():
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    
    #Feature 1
    pa_preds = nx.preferential_attachment(G,future_connections.index)
    jc_preds = nx.jaccard_coefficient(G,future_connections.index)


    pa_my_l = [] # preferntial attachment
    pa_my_val = []

    jc_my_l = [] # jacard attachment
    jc_my_val = []

    my_dfs = [pa_preds, jc_preds]
    count = 0
    for my_df in my_dfs:
        if count == 0:
            for u,v,p in my_df:
                pa_my_l.append((u,v))
                pa_my_val.append(p)
        if count > 0:
            for u,v,p in my_df:
                jc_my_l.append((u,v))
                jc_my_val.append(p)
        count += 1

    # write a code to check if the two nodes are in the same or a different department, if so add either a one or a zero

    pa_preds = nx.preferential_attachment(G,future_connections.index)

    same_dep = []
    for u,v,p in pa_preds:
        if list(G.nodes[u].values())[0] == list(G.nodes[v].values())[0]:
            same_dep.append(1)
            va1 = list(G.nodes[u].values())[0]
            va2 = list(G.nodes[v].values())[0]
            #print(str(va1)+ "and" + str(va2))
        else:
            same_dep.append(0)


    # get the transformed data set 
    df_pa = pd.DataFrame({"pa_score":pa_my_val, "same_depart": same_dep}, index= pa_my_l)
    df_jc = pd.DataFrame({"jc_score":jc_my_val}, index= jc_my_l)            


    df_score = pd.merge(df_jc, df_pa, left_index=True, right_index=True)
    df_fin = pd.merge(future_connections, df_score, left_index=True, right_index=True)

    # machine learning

    # train test set
    test_X = df_fin[df_fin["Future Connection"].isnull()].iloc[:,[1,2,3]]
    train_X = df_fin[df_fin["Future Connection"].notnull()].iloc[:,[0,1,2,3]]
    train_X["Future Connection"] = train_X["Future Connection"].astype(int)


    # going with a logistic regression
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression().fit(train_X.iloc[:,[1,2,3]], train_X.iloc[:,0])
    res = reg.predict(test_X.iloc[:,[0,1,2]])


    fin = reg.predict_proba(test_X.iloc[:,[0,1,2]])
    te = [(x[1]) for x in fin]
    res = pd.Series(te, index= test_X.index.values)

    return res

new_connections_predictions()
