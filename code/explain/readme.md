# Teaching Categories to Human Learners with Visual Explanations - Main Code
Code for generating teaching sequences in multi-class setting. Random, Strict, and Explain are implemented.  

## Notes  
Run `main.py` to execute code.  
It will output teaching strategy files that can be used in the web interface.  
For 2D binary datasets it will plot the hypothesis space.


The following are good settings for visualization:  
```
dataset_name = datasets[1]
do_pca = True
pca_dims = 2
num_init_hyps = 10
num_teaching_itrs = 5
hyp_type = 'rand'
```

When comparing the multi-class model to the binary one make sure to scale alpha appropriately i.e. for the binary case `alpha = alpha/2.0`.

`hyps` is a list of hypothesis where each entry is a CxD matrix. For a binary classification problem a hypothesis = np.vstack((-w, w)), where w is the linear weight vector.  
`X` is the NxD feature matrix.  
`Y` is the N vector of labels. Labels go from 0 to number of classes.   
`explain_interp` is the N vector of explanation interpretability. 1.0 means easy to interpret and 0 means hard. Setting these to all ones will be the same as using vanilla strict.    


## Reference
If you find our work useful in your research please consider citing our paper:  
```
@inproceedings{explainteachcvpr18,
  title     = {Teaching Categories to Human Learners with Visual Explanations},
  author    = {Mac Aodha, Oisin and Su, Shihan and Chen, Yuxin and Perona, Pietro and Yue, Yisong},
  booktitle = {CVPR},
  year = {2018}
}
```
