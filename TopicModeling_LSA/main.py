#%%
import functions as cf

# LSA Model
number_of_topics=7
words=10
document_list,titles=cf.load_data("C:\\MyWorkspace\\python_pj\\TopicModeling_LSA\\articles.txt")
# document_list,titles=cf.load_data("C:\MyWorkspace\python_pj\TopicModeling_LSA","articles.txt")
clean_text=cf.preprocess_data(document_list)

#%%
# plot coherence score values
start,stop,step=2,12,1
cf.plot_graph(clean_text,start,stop,step)

#%%
model=cf.create_gensim_lsa_model(clean_text,number_of_topics,words)