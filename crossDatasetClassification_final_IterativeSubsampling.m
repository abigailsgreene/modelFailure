% Code to run cross-dataset classification with training set = whole
% sample, only WSC, or only WSM
% July 14, 2021
% Abigail Greene

% Initialize some things
numiters = 3; % whole sample model, correct model, misclass model
numiter2 = 100; % number of subsampling iters to do
train = 'dsd'; % which dataset is train vs. test? (yale = 'dsd' vs. 'ucla')
numedges = 500; % number of positively and negatively correlated edges to select (for total #edges in model = 2*numedges)
homedir = '/data13/mri_group/abby_data/misclassificationAnalyses/data/'; % where input data are stored
resultdir = '/data13/mri_group/abby_data/misclassificationAnalyses/results/'; % where outputs should be saved
exclude_outliers = 1; %1=yes, 0=no
permtest = input('perm test?'); % 1 = yes, 0 = no
model_type = 'flip'; % flip the pos/neg corr edges or don't flip ('flip' vs. 'noflip')

% load in DSD matrices and phenotypic data
load([homedir 'DSD/all_mats_meanThresh0.15_maxThresh0.2_allSubs.mat']);
load([homedir 'DSD/crossDataset_phenotypes.mat']);
datamat_dsd = datamat_crossSubset; clear datamat_crossSubset
names_dsd = {'wais_LN_scaled','wasi_vocab_scaled','wasi_mr_scaled'};
% load in UCLA matrices and phenotypic data
load([homedir 'UCLA/ALL_7tasks_wnulls_20210406.mat']);
names_ucla = {'ln_std', 'vocab_std','mr_std'};
[~,~,datamat_tmp] = xlsread([homedir 'UCLA/normedNeuropsych.xlsx']); % load in three scores (WAIS LN, vocab, and MR, normed by AG)
for ucla_np = 1:length(names_ucla)
    col_idx = find(strcmp(datamat_tmp(1,:),names_ucla{ucla_np}));
    datamat_ucla(:,ucla_np) = cell2mat(datamat_tmp(2:end,col_idx));
    clear col_idx
end
clear datamat_tmp

% vectorize imaging data
low_idx = find(tril(ones(size(all_mats,1),size(all_mats,1)),-1));
for i = 1:size(all_mats,3)
    for j = 1:size(all_mats,4)
        tmpmat = squeeze(all_mats(:,:,i,j));
        all_vecs_dsd(:,i,j) = tmpmat(low_idx); % edge x subject x task
        clear tmpmat
    end
end
gfc_vecs_dsd = mean(all_vecs_dsd,3); % GFC = mean GFC across conditions

for i = 1:size(ALL,3)
    for j = 1:size(ALL,4)
        tmpmat = squeeze(ALL(:,:,i,j));
        all_vecs_ucla(:,i,j) = tmpmat(low_idx); % edge x subject x task
        clear tmpmat
    end
end
gfc_vecs_ucla = mean(all_vecs_ucla,3); % GFC = mean GFC across condnitions

% set FC to use
dsd_fcvec = gfc_vecs_dsd;
ucla_fcvec = gfc_vecs_ucla;

% number of subjects
nsubs_dsd = size(datamat_dsd,1);
nsubs_ucla = size(datamat_ucla,1);

% establish high/low thresholds and find corresponding subjects
for i = 1:length(names_dsd)
    if contains(names_dsd{i},{'wasi'}) % t scores (mean 50, std 10)
        stdScore_mean_dsd(i) = 50; stdScore_sd_dsd(i) = 9; % rounded; see CVclassification_standardizedScores_final.m for explanation
    elseif contains(names_dsd{i},{'wais'}) % scaled scores (mean 10, std 3)
        stdScore_mean_dsd(i) = 10; stdScore_sd_dsd(i) = 3;
    end
    stdScore_mean_ucla(i) = 10; stdScore_sd_ucla(i) = 3; % all wais for UCLA, with mean=10,sd=3
    
    % exclude subjects with outlier (>=3sd <mean) phenotypic data based on population norms (i.e., code their data as nan)
    if exclude_outliers==1
        outlier_ct = 0;
        for ii = 1:size(datamat_dsd,1) % rows
            if datamat_dsd(ii,i)<=(stdScore_mean_dsd(i)-(3*stdScore_sd_dsd(i)))
                datamat_dsd(ii,i) = NaN;
                outlier_ct = outlier_ct+1;
            else
            end
        end
        disp(['number of dsd outliers = ' num2str(outlier_ct)]) % this will be counted separately for each phenotypic measure
        outlier_ct = 0;
        for ii = 1:size(datamat_ucla,1) % rows
            if datamat_ucla(ii,i)<=(stdScore_mean_ucla(i)-(3*stdScore_sd_ucla(i)))
                datamat_ucla(ii,i) = NaN;
                outlier_ct = outlier_ct+1;
            else
            end
        end
        disp(['number of ucla outliers = ' num2str(outlier_ct)])
    else
    end
    
    % establish low/high cutoffs and store indices of low and high scorers
    cutoff_low_dsd(i) = stdScore_mean_dsd(i)-((1/3)*stdScore_sd_dsd(i));
    cutoff_high_dsd(i) = stdScore_mean_dsd(i)+((1/3)*stdScore_sd_dsd(i));
    lowScore_dsd_idx{i} = find(datamat_dsd(:,i)<=cutoff_low_dsd(i));
    highScore_dsd_idx{i} = find(datamat_dsd(:,i)>=cutoff_high_dsd(i));
    subset_dsd_idx{i} = sort([lowScore_dsd_idx{i}; highScore_dsd_idx{i}],'ascend');
    
    cutoff_low_ucla(i) = stdScore_mean_ucla(i)-((1/3)*stdScore_sd_ucla(i));
    cutoff_high_ucla(i) = stdScore_mean_ucla(i)+((1/3)*stdScore_sd_ucla(i));
    lowScore_ucla_idx{i} = find(datamat_ucla(:,i)<=cutoff_low_ucla(i));
    highScore_ucla_idx{i} = find(datamat_ucla(:,i)>=cutoff_high_ucla(i));
    subset_ucla_idx{i} = sort([lowScore_ucla_idx{i}; highScore_ucla_idx{i}],'ascend');
    
    % load and select correct and incorrect subject lists
    ucla_result{i} = load([resultdir 'ucla/LOOCVclassification_ucla_bothFC_' names_ucla{i} '_standardScores_100iters_p0.05thresh_permTestfalse_outlierExclusiontrue.mat'],'misclass_subs','correct_subs','binary_subAccMat');
    dsd_result{i} = load([resultdir 'dsd/LOOCVclassification_dsd_bothFC_' names_dsd{i} '_standardScores_100iters_p0.05thresh_permTestfalse_outlierExclusiontrue.mat'],'misclass_subs','correct_subs','binary_subAccMat');
    % find median accuracy for given np measure using GFC
    for iter = 1:size(dsd_result{i}.binary_subAccMat{end},2)
        dsd_result{i}.acc(iter,1) = length(dsd_result{i}.correct_subs{iter,end})/length(find(~isnan(dsd_result{i}.binary_subAccMat{end}(:,iter))));
        ucla_result{i}.acc(iter,1) = length(ucla_result{i}.correct_subs{iter,end})/length(find(~isnan(ucla_result{i}.binary_subAccMat{end}(:,iter))));
    end
    dsdmedian_idx = find(dsd_result{i}.acc==median(dsd_result{i}.acc));
    % if median is average of two middle values and thus doesn't exist as single iteration's accuracy:
    if isempty(dsdmedian_idx)
        [~,accidx_dsd] = sort(dsd_result{i}.acc,'ascend');
        dsdmedian_idx = find(dsd_result{i}.acc==dsd_result{i}.acc(accidx_dsd((length(accidx_dsd)/2))));
    end
    uclamedian_idx = find(ucla_result{i}.acc==median(ucla_result{i}.acc));
    if isempty(uclamedian_idx)
        [~,accidx_ucla] = sort(ucla_result{i}.acc,'ascend');
        uclamedian_idx = find(ucla_result{i}.acc==ucla_result{i}.acc(accidx_ucla((length(accidx_ucla)/2))));
    end
    % or if more than one iteration == median value
    if length(dsdmedian_idx)>1
        [~,best_idx] = max(corr(dsd_result{i}.binary_subAccMat{end}(:,dsdmedian_idx),mean(dsd_result{i}.binary_subAccMat{end},2),'rows','complete')); % pick first iteration where misclassified subject set is most similar to (ie correlated with) overall misclassification frequency for GFC/that NP measure
        dsd_medianAcc_iter(i) = dsdmedian_idx(best_idx);
        clear best_idx
    elseif length(dsdmedian_idx)==1
        dsd_medianAcc_iter(i) = dsdmedian_idx;
    end
    if length(uclamedian_idx)>1
        [~,best_idx] = max(corr(ucla_result{i}.binary_subAccMat{end}(:,uclamedian_idx),mean(ucla_result{i}.binary_subAccMat{end},2),'rows','complete')); % pick first iteration where misclassified subject set is most similar to (ie correlated with) overall misclassification frequency for GFC/that NP measure
        ucla_medianAcc_iter(i) = uclamedian_idx(best_idx);
        clear best_idx
    elseif length(uclamedian_idx)==1
        ucla_medianAcc_iter(i) = uclamedian_idx;
    end
    
    % binarize and subset phenotypes (GFC: tot, correct and misclassified
    % subjects from iteration with median accuracy)
    binarized_phen_dsd(:,i) = zeros(nsubs_dsd,1);
    binarized_phen_dsd(lowScore_dsd_idx{i},i) = -1;
    binarized_phen_dsd(highScore_dsd_idx{i},i) = 1;
    binarized_phen_ucla(:,i) = zeros(nsubs_ucla,1);
    binarized_phen_ucla(lowScore_ucla_idx{i},i) = -1;
    binarized_phen_ucla(highScore_ucla_idx{i},i) = 1;
    
    binarized_phen_dsd_correct{i} = binarized_phen_dsd(dsd_result{i}.correct_subs{dsd_medianAcc_iter(i),end},i);
    binarized_phen_dsd_misclass{i} = binarized_phen_dsd(dsd_result{i}.misclass_subs{dsd_medianAcc_iter(i),end},i);
    binarized_phen_dsd_tot{i} = binarized_phen_dsd(subset_dsd_idx{i},i);
    binarized_phen_ucla_correct{i} = binarized_phen_ucla(ucla_result{i}.correct_subs{ucla_medianAcc_iter(i),end},i);
    binarized_phen_ucla_misclass{i} = binarized_phen_ucla(ucla_result{i}.misclass_subs{ucla_medianAcc_iter(i),end},i);
    binarized_phen_ucla_tot{i} = binarized_phen_ucla(subset_ucla_idx{i},i);
    % check to make sure we're not including any subs with intermediate
    % or missing memory score (should not be the case given that
    % correct/misclass labels come from within-sample classification)
    disp(['length of correct dsd subs with binarized phen = 0 is ' num2str(length(find(binarized_phen_dsd_correct{i}==0))) '; length of misclass dsd subs with binarized phen = 0 is ' num2str(length(find(binarized_phen_dsd_misclass{i}==0)))]);
    disp(['length of correct ucla subs with binarized phen = 0 is ' num2str(length(find(binarized_phen_ucla_correct{i}==0))) '; length of misclass ucla subs with binarized phen = 0 is ' num2str(length(find(binarized_phen_ucla_misclass{i}==0)))]);
    
    % subset FC (tot, correct, and misclassified subsets)
    dsd_fcvec_correct{i} = dsd_fcvec(:,dsd_result{i}.correct_subs{dsd_medianAcc_iter(i),end});
    dsd_fcvec_misclass{i} = dsd_fcvec(:,dsd_result{i}.misclass_subs{dsd_medianAcc_iter(i),end});
    dsd_fcvec_tot{i} = dsd_fcvec(:,subset_dsd_idx{i});
    ucla_fcvec_correct{i} = ucla_fcvec(:,ucla_result{i}.correct_subs{ucla_medianAcc_iter(i),end});
    ucla_fcvec_misclass{i} = ucla_fcvec(:,ucla_result{i}.misclass_subs{ucla_medianAcc_iter(i),end});
    ucla_fcvec_tot{i} = ucla_fcvec(:,subset_ucla_idx{i});
end

% ok, so now we have three versions of FC (all low+high subs, only
% previously correctly classified subs, only previously incorrectly
% classified subs) and phenotypic scores to go along with each
for phen_idx = 1:length(names_dsd)
    clearvars -except permtest idx_perm_* model_type names_* phen_idx dsd_fcvec* ucla_fcvec* binarized_phen* subset_* numiter* use_gfc train numedges exclude_outliers ucla_result dsd_result *medianAcc_iter homedir resultdir
    for iter = 1:numiters % 3 iters: all, WSC, WSM
        clear train_mats train_phen* test_mats test_phen*
        % if training with ucla
        if iter==1 && strcmp(train,'ucla') % train = all
            test_mats = dsd_fcvec_tot{phen_idx};
            test_phen = binarized_phen_dsd_tot{phen_idx};
            train_mats = ucla_fcvec_tot{phen_idx};
            train_phen = binarized_phen_ucla_tot{phen_idx};
        elseif iter==2 && strcmp(train,'ucla') % train = WSC
            test_mats = dsd_fcvec_tot{phen_idx};
            test_phen = binarized_phen_dsd_tot{phen_idx};
            train_mats = ucla_fcvec_correct{phen_idx};
            train_phen = binarized_phen_ucla_correct{phen_idx};
        elseif iter==3 && strcmp(train,'ucla') % train = WSM
            test_mats = dsd_fcvec_tot{phen_idx};
            test_phen = binarized_phen_dsd_tot{phen_idx};
            train_mats = ucla_fcvec_misclass{phen_idx};
            train_phen = binarized_phen_ucla_misclass{phen_idx};
            % if training with yale and testing on ucla
        elseif iter==1 && strcmp(train,'dsd')
            train_mats = dsd_fcvec_tot{phen_idx};
            train_phen = binarized_phen_dsd_tot{phen_idx};
            test_mats = ucla_fcvec_tot{phen_idx};
            test_phen = binarized_phen_ucla_tot{phen_idx};
        elseif iter==2 && strcmp(train,'dsd')
            train_mats = dsd_fcvec_correct{phen_idx};
            train_phen = binarized_phen_dsd_correct{phen_idx};
            test_mats = ucla_fcvec_tot{phen_idx};
            test_phen = binarized_phen_ucla_tot{phen_idx};
        elseif iter==3 && strcmp(train,'dsd')
            train_mats = dsd_fcvec_misclass{phen_idx};
            train_phen = binarized_phen_dsd_misclass{phen_idx};
            test_mats = ucla_fcvec_tot{phen_idx};
            test_phen = binarized_phen_ucla_tot{phen_idx};
        end
        
        edges_pos{iter} = zeros(size(train_mats,1),numiter2); edges_neg{iter} = zeros(size(train_mats,1),numiter2);% initialize arrays to store selected edges
        train_phen_real = train_phen; test_phen_real = test_phen;
        % If permutation testing, generate permutations
        if permtest==1
            for iter2=1:numiter2
                idx_perm_train{phen_idx,iter}(:,iter2) = randperm(length(train_phen));
                while (iter2~=1 && any(sum(abs(idx_perm_train{phen_idx,iter}(:,1:iter2-1)-idx_perm_train{phen_idx,iter}(:,iter2)))==0)) || isequal(idx_perm_train{phen_idx,iter}(:,iter2),[1:length(train_phen)]') % if any of the columns are repeated or permuted indices = true indices, regenerate permuted order
                    idx_perm_train{phen_idx,iter}(:,iter2) = randperm(length(train_phen));
                end
                
                idx_perm_test{phen_idx,iter}(:,iter2) = randperm(length(test_phen));
                while (iter2~=1 && any(sum(abs(idx_perm_test{phen_idx,iter}(:,1:iter2-1)-idx_perm_test{phen_idx,iter}(:,iter2)))==0)) || isequal(idx_perm_test{phen_idx,iter}(:,iter2),[1:length(test_phen)]') % if any of the columns are repeated or permuted indices = true indices, regenerate permuted order
                    idx_perm_test{phen_idx,iter}(:,iter2) = randperm(length(test_phen));
                end
            end
        else
        end
        
        for iter2 = 1:numiter2
            if permtest==1 % permute all phenotypic scores
                clear train_phen test_phen
                train_phen = train_phen_real(idx_perm_train{phen_idx,iter}(:,iter2));
                test_phen = test_phen_real(idx_perm_test{phen_idx,iter}(:,iter2));
            else
            end
            
            % subsample to ensure equal sized classes in training sample
            train_idx_low = find(train_phen==-1);
            train_idx_high = find(train_phen==1);
            % which is bigger class?
            disp(['length pre-sampling low score = ' num2str(length(train_idx_low))]);
            disp(['length pre-sampling high score = ' num2str(length(train_idx_high))]);
            num_subs_subset = min([length(train_idx_low) length(train_idx_high)]);
            if length(train_idx_low)>length(train_idx_high)
                train_idx_low = train_idx_low(randperm(length(train_idx_low),num_subs_subset));
            elseif length(train_idx_low)<length(train_idx_high)
                train_idx_high = train_idx_high(randperm(length(train_idx_high),num_subs_subset));
            else disp('low and high groups are the same size')
            end
            % sanity check to ensure classes are now the same size
            disp(['length post-sampling low score = ' num2str(length(train_idx_low))]);
            disp(['length post-sampling high score = ' num2str(length(train_idx_high))]);
            
            % subsample FC and phenotypic data
            train_subset = sort([train_idx_low; train_idx_high],'ascend');
            train_mats_subsample = train_mats(:,train_subset);
            train_phen_subsample = train_phen(train_subset);
            train_subset_save{iter}(:,iter2) = train_subset; % save out subset idx 
            
            % ensure subsamping is unique on each iter2
            while iter2~=1 && any(sum(abs(train_subset_save{iter}(:,1:iter2-1)-train_subset_save{iter}(:,iter2)))==0) % if subset is same as on any previous iteration, redo subsetting
                clear train_idx_low train_idx_high train_subset train_mats_subsample train_phen_subsample
                train_subset_save{iter}(:,iter2) = [];
                train_idx_low = find(train_phen==-1);
                train_idx_high = find(train_phen==1);
                num_subs_subset = min([length(train_idx_low) length(train_idx_high)]);
                if length(train_idx_low)>length(train_idx_high)
                    train_idx_low = train_idx_low(randperm(length(train_idx_low),num_subs_subset));
                elseif length(train_idx_low)<length(train_idx_high)
                    train_idx_high = train_idx_high(randperm(length(train_idx_high),num_subs_subset));
                else disp('low and high groups are the same size')
                end
                train_subset = sort([train_idx_low; train_idx_high],'ascend');
                train_mats_subsample = train_mats(:,train_subset);
                train_phen_subsample = train_phen(train_subset);
                train_subset_save{iter}(:,iter2) = train_subset;
            end
            
            % find correlated edges using sparsity threshold
            [r,p] = corr(train_mats_subsample',train_phen_subsample);
            [~,rsortidx] = sort(r,'descend');
            disp(['length of r pos = ' num2str(length(find(r>0))) ', lowest selected rpos = ' num2str(r(rsortidx(numedges)))]);
            disp(['length of r neg = ' num2str(length(find(r<0))) ', highest selected rneg = ' num2str(r(rsortidx(end-(numedges-1))))]);
            
            edges_pos{iter}(rsortidx(1:numedges),iter2) = 1;
            edges_neg{iter}(rsortidx(end-(numedges-1):end),iter2) = 1;
            
            % sum across selected edges
            for sub = 1:size(train_mats_subsample,2)
                train_edgeScore{iter2,iter}(sub,:) = [sum(train_mats_subsample(find(edges_pos{iter}(:,iter2)), sub)) sum(train_mats_subsample(find(edges_neg{iter}(:,iter2)), sub))];
            end
            
            % train model
            trainmodel{iter2,iter} = fitcsvm(zscore(train_edgeScore{iter2,iter}),train_phen_subsample,'KernelFunction','linear');
            
            % apply model to test subjects
            for sub = 1:size(test_mats,2) 
                if strcmp(model_type,'noflip')
                    test_edgeScore{iter2,iter}(sub,:) =  [sum(test_mats(find(edges_pos{iter}(:,iter2)), sub)) sum(test_mats(find(edges_neg{iter}(:,iter2)), sub))];
                elseif strcmp(model_type,'flip')
                    test_edgeScore{iter2,iter}(sub,:) =  [sum(test_mats(find(edges_neg{iter}(:,iter2)), sub)) sum(test_mats(find(edges_pos{iter}(:,iter2)), sub))];
                end
            end
            pred_phen_svm{iter2,iter} = predict(trainmodel{iter2,iter},zscore(test_edgeScore{iter2,iter}));
            
            % calculate accuracy in various ways
            if strcmp(train,'dsd')
                thigh_phigh{iter2,iter} = subset_ucla_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==1),find(test_phen==1)));
                thigh_plow{iter2,iter} = subset_ucla_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==-1),find(test_phen==1)));
                tlow_phigh{iter2,iter} = subset_ucla_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==1),find(test_phen==-1)));
                tlow_plow{iter2,iter} = subset_ucla_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==-1),find(test_phen==-1)));
            elseif strcmp(train,'ucla')
                thigh_phigh{iter2,iter} = subset_dsd_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==1),find(test_phen==1)));
                thigh_plow{iter2,iter} = subset_dsd_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==-1),find(test_phen==1)));
                tlow_phigh{iter2,iter} = subset_dsd_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==1),find(test_phen==-1)));
                tlow_plow{iter2,iter} = subset_dsd_idx{phen_idx}(intersect(find(pred_phen_svm{iter2,iter}==-1),find(test_phen==-1)));
            else
                disp('problem. what is training data?')
            end
            % overall accuracy
            acc_tot(iter2,iter) = (length(thigh_phigh{iter2,iter})+length(tlow_plow{iter2,iter}))/size(test_phen,1);
            % accuracy for WSC and WSM separately
            if strcmp(train,'ucla')
                correctSub_acc(iter2,iter) = (length(find(ismember(dsd_result{phen_idx}.correct_subs{dsd_medianAcc_iter(phen_idx),end},thigh_phigh{iter2,iter})))+length(find(ismember(dsd_result{phen_idx}.correct_subs{dsd_medianAcc_iter(phen_idx),end},tlow_plow{iter2,iter}))))/length(dsd_result{phen_idx}.correct_subs{dsd_medianAcc_iter(phen_idx),end});
                misclassSub_acc(iter2,iter) = (length(find(ismember(dsd_result{phen_idx}.misclass_subs{dsd_medianAcc_iter(phen_idx),end},thigh_phigh{iter2,iter})))+length(find(ismember(dsd_result{phen_idx}.misclass_subs{dsd_medianAcc_iter(phen_idx),end},tlow_plow{iter2,iter}))))/length(dsd_result{phen_idx}.misclass_subs{dsd_medianAcc_iter(phen_idx),end});
            elseif strcmp(train,'dsd')
                correctSub_acc(iter2,iter) = (length(find(ismember(ucla_result{phen_idx}.correct_subs{ucla_medianAcc_iter(phen_idx),end},thigh_phigh{iter2,iter})))+length(find(ismember(ucla_result{phen_idx}.correct_subs{ucla_medianAcc_iter(phen_idx),end},tlow_plow{iter2,iter}))))/length(ucla_result{phen_idx}.correct_subs{ucla_medianAcc_iter(phen_idx),end});
                misclassSub_acc(iter2,iter) = (length(find(ismember(ucla_result{phen_idx}.misclass_subs{ucla_medianAcc_iter(phen_idx),end},thigh_phigh{iter2,iter})))+length(find(ismember(ucla_result{phen_idx}.misclass_subs{ucla_medianAcc_iter(phen_idx),end},tlow_plow{iter2,iter}))))/length(ucla_result{phen_idx}.misclass_subs{ucla_medianAcc_iter(phen_idx),end});
            end
            clearvars -except permtest idx_perm_* model_type train_mats train_phen train_phen_real test_mats test_phen test_phen_real train_subset_save names_* dsd_fcvec* ucla_fcvec* binarized_phen* subset_* thigh* tlow* numiter* use_gfc train numedges exclude_outliers ucla_result dsd_result *medianAcc_iter *Sub_acc acc_tot pred_* iter phen_idx edges_* *_edgeScore betas* trainmodel homedir resultdir
        end
    end
    % save out results
    save(strjoin([resultdir 'crossDataset/' train 'Train_' num2str(numedges) 'edges_' names_ucla{phen_idx} '_' num2str(numiter2) 'itersSubsampling_PosNegEdges' model_type '_permTest' string(logical(permtest)) '.mat'],''));
end