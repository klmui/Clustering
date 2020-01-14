#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
def EM_step_1(points, mu, sigma):
    """Computes the density values of each point (E-step part 1).
    
    Args:
        points: set of points.
        mu: list of cluster means.
        sigma: spread of the distribution.
    
    """
    f_dict = {}
    for i in range(0, len(mu)):
        cluster_mean = mu[i]
        for j in range(0, len(points)):
            point = points[j]
            gaussian_density = (1 / (sigma * math.sqrt(2*math.pi))) * math.exp(-1/2 * (((point - cluster_mean) / sigma)**2))
            f_dict[(point, i + 1)] = gaussian_density
    return f_dict


# In[2]:


def EM_step_2(points, mu, f_dict, prior_probs):
    """Computes the probabilities of each point being assigned to each cluster.
    
    Args:
        points: set of points.
        mu: list of cluster means.
        f_dict: density dictionary.
        prior_probs: list of prior_probabilities.
    
    """
    h_dict = {}
    point_count = 1
    for i in range(0, len(points)):
        point = points[i]
        for j in range(0, len(mu)):
                cluster_mean = mu[j]
                h_val_numerator = prior_probs[j] * f_dict[(point, j + 1)]
                h_val_denominator = (prior_probs[0] * f_dict[(point, 1)]) + (prior_probs[1] * f_dict[(point, 2)])
                h_dict[(point_count, j + 1)] = h_val_numerator / h_val_denominator
        point_count += 1
    return h_dict


# In[3]:


def EM_m_step(mu, prior_probs, points, h_dict):
    """Updates parameters with expected values calculated.
    
    Args:
        mu: list of cluster means.
        prior_probs: list of prior_probabilities.
        points: set of points.
        h_dict: dictionary with probabilities of each point being assigned to each cluster.
    
    """
    updated_mu = []
    updated_prior_probs = []
    # Get updated cluster means
    for i in range(0, len(mu)):
        new_mu_numerator = 0
        new_mu_denominator = 0
        for j in range(0, len(points)):
            point = points[j]
            new_mu_numerator += point * max(h_dict[j + 1, i + 1], h_dict[j + 1, i + 1])
            new_mu_denominator += max(h_dict[j + 1, i + 1], h_dict[j + 1, i + 1])
        updated_mu.append(new_mu_numerator / new_mu_denominator)
        
    # Get updated prior probabilities
    new_prior_prob_denominator = len(points)
    for i in range(0, len(prior_probs)):
        new_prior_prob_numerator = 0
        for j in range(0, len(points)):
            new_prior_prob_numerator += max(h_dict[j + 1, i + 1], h_dict[j + 1, i + 1])
        updated_prior_probs.append(new_prior_prob_numerator / new_prior_prob_denominator)
        
    updated_mu_and_prior_probs = []
    updated_mu_and_prior_probs.append(updated_mu)
    updated_mu_and_prior_probs.append(updated_prior_probs)
    return updated_mu_and_prior_probs


# In[4]:


def EM_algorithm_Gaussian(mu, sigma, points, prior_probs, iterations):
    mu = mu
    prior_probs = prior_probs
    for i in range(0, iterations):
        print("==============================================================================================================")
        print("Iteration:", i + 1)
        
        # Get density values (E-step)
        f_dict = EM_step_1(points, mu, sigma)
        print("Density values:")
        for key, value in f_dict.items():
            cluster_number = str(key[1])
            cluster = "mu_" + cluster_number
            print("f(" + str(key[0]) + "|" + cluster + ") =", value)
        print("\n")
        
        # Get probabilities of each point being assigned to each cluster (E-step)
        h_dict = EM_step_2(points, mu, f_dict, prior_probs)
        print("(i) Probabilities of each point being assigned to each cluster:")
        for key, value in h_dict.items():
            cluster_number = str(key[1])
            cluster = "mu_" + cluster_number
            print("h_" + str(key) + " = " + str(value), sep="")
        print("\n")
        
        # Get updated cluster means and updated cluster prior probabilities (M-step)
        updated_mu_and_prior_probs = EM_m_step(mu, prior_probs, points, h_dict)
        mu = updated_mu_and_prior_probs[0]
        print("(ii) Updated cluster means:")
        for j in range(0, len(mu)):
            print("mu_" + str(j + 1) + " =", mu[j])
        print("\n")
        prior_probs = updated_mu_and_prior_probs[1]
        print("(iii) Updated cluster prior probabilities:")
        for z in range(0, len(prior_probs)):
            print("P_" + str(z + 1) + " =", prior_probs[z])
            
        print("End of iteration:", i + 1)
        print("============================================================================================================== \n")


# In[7]:


# Example where mu_1 = 3 and mu_2 = 2. Initial cluster prior probabilities P_1 and P_2 = 0,5. The variances sigma_1 squared and sigma_2 squared = 3.
EM_algorithm_Gaussian([3, 2], math.sqrt(3), [2, 4, 5, 9, 10], [0.5, 0.5], 3)


# In[ ]:




