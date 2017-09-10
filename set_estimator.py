import numpy as np 
import itertools, copy

class SetEstimator:
    """
    Construct a right-stochastic transition matrix that describes the probability of
    making an observation in each region of the venn diagram given that the 
    system is really in any other region.  If you have categories A and B you will
    have the following regions in the venn diagram:

        "-" (neither A nor B)
        "A" (A alone)
        "B" (B alone)
        "AB" (in A and B).

    The transition matrix will be:

    P(-_obs|-_real), P(A_obs|-_real), P(B_obs|-_real), P(AB_obs|-_real)
    P(-_obs|A_real), P(A_obs|A_real), P(B_obs|A_real), P(AB_obs|A_real)
    P(-_obs|B_real), P(A_obs|B_real), P(B_obs|B_real), P(AB_obs|B_real)
    P(-_obs|AB_real),P(A_obs|AB_real),P(B_obs|AB_real),P(AB_obs|AB_real)

    The matrix is using the false negative and false positive rates for each 
    category call.  
    """
    
    def __init__(self,count_dict,fp,fn):

        if set([type(fp),type(fn),type(count_dict)]) != set([dict]):
            err = "fp, fn, and count_dict must be dictionaries.\n"
            raise ValueError(err)
        
        # Grab false positive and false negative dics 
        self._fp = copy.copy(fp)
        self._fn = copy.copy(fn)
        self._count_dict = copy.copy(count_dict)
        
        # Load the user inputs, doing error checking
        self._load_data()
        
        self._create_transition_matrix()
        
    def _load_data(self):
        """
        Take the dictionaries specified by the user and create an internal data
        structure keeping track of each category and region.  Do error checking
        to make sure that all regions implied by categories have acounts.
        """

        # Get states from dictionaries -- they need to have the same states
        fp_states = set(list(self._fp.keys()))
        fn_states = set(list(self._fn.keys()))

        if fp_states != fn_states:
            err = "False positive and false negative dictionaries must have identical states.\n"
            raise ValueError(err)

        # List of categories
        self._categories = list(fp_states)
        self._categories.sort()
        
        # Create a list of possible regions in the venn diagram.  For example,
        # consider two categories: A and B.  The regions would be A, B, AB and
        # "" (not anywhere in A or B). The region "A" means A *only* and thus
        # excludes *AB*.  
        self._venn = []
        for i in range(len(self._categories)+1):
            for region in itertools.combinations(self._categories,i):
                self._venn.append(region)
        
        regions_not_seen = copy.copy(self.venn_regions)

        tmp_count_dict = {}
        for c in self._count_dict.keys():
            
            # Make sure that keys are in the right format
            new_c = list(c)
            new_c.sort()
            new_c = tuple(c)

            if c == "-":
                new_c = tuple()
                
            if new_c not in self.venn_regions:
                err = "venn region {} not recognized.\n".format(c)
                raise ValueError(err)
            
            regions_not_seen.remove(new_c)
            
            tmp_count_dict[new_c] = self._count_dict[c]
        
        if len(regions_not_seen) != 0:
            if tuple() in regions_not_seen:
                regions_not_seen.remove(tuple())
                regions_not_seen.append("-")
                
            pretty_missing_regions = ["    \"{}\"".format("".join(r)) for r in regions_not_seen]
            pretty_missing_regions = "\n".join(pretty_missing_regions)            
            
            err = "Some venn regions not in count_dict. Missing regions: \n{}".format(pretty_missing_regions)
            raise ValueError(err)
        
        self._count_dict = copy.copy(tmp_count_dict)

        # Create structures for estimator
        self._est_count_dict = copy.copy(self._count_dict)
        self._est_fp = copy.copy(self._fp)
        self._est_fn = copy.copy(self._fn)

        self._obs_count_vector = np.array([self._count_dict[s] for s in self.venn_regions],dtype=float)

    def _create_transition_matrix(self):
        """
        Construct a right-stochastic transition matrix that describes the probability of
        making an observation in each region of the venn diagram given that the 
        system is really in any other region.  
        """
        
        # Final transition matrix we'll build determining probability of
        # from any region to any other region of the venn diagram
        self._trans_mat = np.zeros((len(self._venn),len(self._venn)),dtype=float)

        for i in range(len(self._venn)):

            # We'll be calculating transitions away from this real_region
            real_region = set(self._venn[i])

            # microscopic_prob determines the sign of transitions away from the
            # real region.  If a category is seen in this real region, the
            # transition will be a false negative.  (If we're region AB, all
            # transitions involving category "A" will be losing A -- a false
            # negative). If a category is *not* seen in this real region, the 
            # transition will be a false positive.  (If we're in region B, all
            # transitions involving "A" will be gaining A -- a false positive).
            microscopic_prob = {}
            for c in self._categories:
                if c in real_region:
                    microscopic_prob[c] = self._est_fn[c]
                else:
                    microscopic_prob[c] = self._est_fp[c]

            # Now walk through all regions of the venn diagram and determine the
            # probability of making an observation in "observed_region" given that
            # we are actually in "real_region".
            for j in range(len(self._venn)):

                # Region we observe we're in after false negatives and false positives
                # accumulate
                observed_region = set(self._venn[j])

                # Find differences between observed and real state
                differences = real_region^observed_region   
                num_diff = len(differences)

                #print(real_region,"-->",observed_region,num_diff)

                # Now figure out probability of this particular set of false
                # positives and false negatives being observed. This is a *nasty* 
                # loop that does region intersection calculations to an arbitrary
                # order, accounting appropriately for the fact you have to add and
                # subtract various regions while doing so to avoid over and under
                # counting. 
                prob = 0        
                for k in range(len(self._categories) + 1 - num_diff):

                    combo_order = len(self._categories) - k

                    # This obscure line of code flips the sign appropriately for each overlap 
                    # order to avoid over and undercounting the same regions.  Really, it
                    # does.  
                    sign = ((-1)**(combo_order % 2))*((-1)**(num_diff % 2))
                    
                    # Build terms describing probability of each sub-region within the regions
                    # we're intersecting, adding them to the total probability.  Basically,
                    # this builds sums like:                    
                    # P(in_region_A) = P(in_category_A) - P(in_cateogry_A)*P(in_category_B)
                    # up to any order of overlap.  
                    for combo in itertools.combinations(self._categories,combo_order): 

                        # Skip combinations that do not include the categories that are different                
                        if not differences.issubset(combo):
                            continue

                        local_prob = 1.0
                        for c in combo:
                            local_prob *= microscopic_prob[c]
                                                    
                        prob += sign*local_prob

                        #print(sign,combo,local_prob)
        
                self._trans_mat[i,j] = prob

                #print("prob:",trans_mat[i,j])
                #print("")

            # Force transition matrix row to really add to one. Do so by adding any error
            # back to the i->i transition.  (The row adds mathematically; it might not, 
            # numerically).  Warn if the error is bigger than a tiny numerical problem.  
            err = 1 - sum(self._trans_mat[i,:])
            if abs(err) > 1e-10:
                warnings.warn("Transition matrix row did not sum exactly to one.  Possible numerical or programming error.")
            self._trans_mat[i,i] = self._trans_mat[i,i] + err
            
        # Invert the matrix -- this will be used to estimate the real size of each
        # region given a set of observed sizes
        self._inv_trans_mat = np.linalg.inv(self._trans_mat)
  
   
    def model(self,param):
        """
        Model that can be fed into a regression/sampling tool.  It takes an
        array of floats, maps these back to the underlyihng model, rebuilds
        the transition matrix, and then returns the counts that would be 
        observed given these parameters. The order of the "param" array should
        be: 

        counts in each venn region (in order in self.venn_regions)
        fp rates (in order in categories)
        fn rates (in order in categories)
        """
    
        counter = 0

        # Load estimated counts for each venn region
        for v in self.venn_regions:
            self._est_count_dict[v] = param[counter]
            counter += 1

        rebuild_trans_matrix = False

        # Load false positive rate for each category
        for c in self.categories:

            if param[counter] != self._fp[c]:
                rebuild_trans_matrix = True

            self._est_fp[c] = param[counter]
            counter += 1

        # Load false negative rate for each category
        for c in self.categories:

            if param[counter] != self._fn[c]:
                rebuild_trans_matrix = True

            self._est_fn[c] = param[counter]
            counter += 1

        # only rebuild if a false positive or false negative rate changed
        if rebuild_trans_matrix:
            self._create_transition_matrix()

        # Do prediction
        c = np.array([self._est_count_dict[v] for v in self.venn_regions],
                     dtype=float)

        prediction = np.dot(c,self._trans_mat)

        return prediction
         

    def pretty_trans_mat(self):
        """
        Print out the transition matrix in a human-readable form.
        """
        
        out = []
        for i in range(len(self._trans_mat)):
            for j in range(len(self._trans_mat)):
                out.append("{:10s} -> {:10s}: {:10.3f}\n".format("".join(self._venn[i]),
                                                                 "".join(self._venn[j]),
                                                                 self._trans_mat[i,j]))
        print("".join(out))
                
    def estimate_freqs(self):
        
        estimated_counts = np.dot(self._inv_trans_mat,self._obs_count_vector)
        
        for i in range(len(estimated_counts)):
            print("{:5s} {:10.3f}".format("".join(self.venn_regions[i]),estimated_counts[i]))
        
    @property
    def fn(self):
        return self._fn
    
    @property
    def fp(self):
        return self._fp
    
    @property
    def venn_regions(self):
        """
        List of regions in the venn diagram.  Stable sort order.
        """
        return self._venn
    
    @property
    def categories(self):
        """
        List of categories that make up venn diagram.  Stable sort order.
        """

        return self._categories
