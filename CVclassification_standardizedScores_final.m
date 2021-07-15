%% code to run classification with CV
% July 14, 2021
% Abigail Greene

% initialize some things
seed = randi(100,1);
rng(seed);
permtest=input('permtest?'); %yes=1 or no=0?
exclude_forCoverage=input('exclude for coverage?'); %yes=1 or no=0 to exclude nodes for some % missing
if exclude_forCoverage==1
   pct_missing=input('pct missing? 25, 50, or 75?');
else pct_missing=100; % if not excluding for missing coverage, then just excluding subjects with complete (ie 100% of) node missing
end
numiters=100;
exclude_outliers = 1; %yes=1 or no=0
use_tasks = 'both'; % options: all (individually), average (GFC), or both (append GFC to fourth dimension of FC mat)
dataset = 'dsd'; % dsd or ucla
homedir = ['/data13/mri_group/abby_data/misclassificationAnalyses/data/' upper(dataset) '/']; % directory where input data are stored
savedir = ['/data13/mri_group/abby_data/misclassificationAnalyses/results/']; % directory where outputs should be saved

if strcmp(dataset,'dsd')
    % load table of post-scan scores
    load([homedir 'normedNeuropsych_dsd.mat']); % names = measures, in order; datamat = subject x measure matrix of normed scores
    % set norms
    for i = 1:length(names)
        if contains(names{i},'wrat') % IQ score (mean 100, std 15)
            stdScore_mean(i) = 100; stdScore_sd(i) = 15;
        elseif contains(names{i},{'brief','wasi'}) % t scores (mean 50, std 10)
            stdScore_mean(i) = 50; stdScore_sd(i) = 9; % note true sd = 10, but 10/3 with rounding sets cutoff at <=46 and >=54; cutoffs closer to +/-1/3 sd from the mean are <=47 and >=53, so will set sd=9 to get these cutoffs; does not affect outlier exclusions in our dataset
        elseif contains(names{i},{'wraml','wais','dkefs'}) % scaled scores (mean 10, std 3)
            stdScore_mean(i) = 10; stdScore_sd(i) = 3;
        elseif contains(names{i},{'bnt'}) % z scores
            stdScore_mean(i) = 0; stdScore_sd(i) = 1;
        end
    end
    % load FC matrices
    load([homedir 'all_mats_meanThresh0.15_maxThresh0.2_allSubs.mat']);
        
elseif strcmp(dataset,'ucla')
    % load FC matrices
    load([homedir 'ALL_7tasks_wnulls_20210406.mat']);
    all_mats = ALL; clear ALL
    sublist_tot=list;
    % load in behavior to predict (datamat) and set population norms
    names = {'ln_std', 'vocab_std','mr_std'}; % same order as in DSD
    stdScore_mean = [10 10 10]; stdScore_sd = [3 3 3]; % WAIS = scaled scores
    [~,~,datamat_tmp] = xlsread([homedir 'normedNeuropsych.xlsx']); % load in three scores (WAIS LN, vocab, and MR, normed by ASG)
    for ucla_np = 1:length(names)
        col_idx = find(strcmp(datamat_tmp(1,:),names{ucla_np}));
        datamat(:,ucla_np) = cell2mat(datamat_tmp(2:end,col_idx));
    end
end

% exclude subjects who are missing XX% of any node if excluding for
% coverage
if exclude_forCoverage==1
    load([homedir 'badSubIDX_missingCoverage_' dataset '.mat']);
    if pct_missing==25
        to_exclude = badSubIDX_25pctMissing;
    elseif pct_missing==50
        to_exclude = badSubIDX_50pctMissing;
    elseif pct_missing==75
        to_exclude = badSubIDX_75pctMissing;
    else disp('error: unknown node coverage requirement');   
    end
    disp(['excluding ' num2str(length(to_exclude)) ' subjects missing ' num2str(pct_missing) '% of >=1 node']);
    datamat(to_exclude,:) = [];
    all_mats(:,:,to_exclude,:) = [];
else
end

% exclude subjects with outlier (>=3sd <mean) phenotypic data based on population norms (i.e., code their data as nan)
if exclude_outliers==1
    outlier_ct = 0;
    for i = 1:size(datamat,1) % rows
        for j = 1:size(datamat,2) % cols
            if datamat(i,j)<=(stdScore_mean(j)-(3*stdScore_sd(j)))
                datamat(i,j) = NaN;
                outlier_ct = outlier_ct+1;
            else
            end
        end
    end
    disp(['number of outliers = ' num2str(outlier_ct)])
else
end

% vectorize imaging data
low_idx = find(tril(ones(size(all_mats,1),size(all_mats,1)),-1));
for i = 1:size(all_mats,3)
    for j = 1:size(all_mats,4)
        tmpmat = squeeze(all_mats(:,:,i,j));
        all_vecs(:,i,j) = tmpmat(low_idx); % edge x subject x task
        clear tmpmat
    end
end
gfc_vecs = mean(all_vecs,3); % GFC = average FC across conditions for each subject
    
% what kind of matrices are we using?
if strcmp(use_tasks,'all')
    mats_toUse = all_vecs;
elseif strcmp(use_tasks,'average')
    mats_toUse = gfc_vecs;
elseif strcmp(use_tasks,'both')
    mats_toUse = cat(3,all_vecs,gfc_vecs);
    disp(['size of mats is ' num2str(size(mats_toUse))]);
else error('unknown matrix data type')
end

% set some other params
task_idx_max = size(mats_toUse,3); % number of FC conditions
phen_idx_max = size(datamat,2); % number of phenotypic measures
nsubs = size(mats_toUse,2); % number of subjects
if permtest==1
    np_highlow = cell(phen_idx_max,1);
    idx_perm = cell(phen_idx_max,1);
end

for phen_idx = 1:phen_idx_max
    % clear everything except matrices, phenotypic data, sub/phenotype lists, and params
    clearvars -except pct_missing savedir phen_idx mats_toUse datamat* sublist* names stdScore_* permtest numiters exclude_outliers seed use_tasks *_idx_max nsubs dataset np_highlow idx_perm
    
    % initialize some things
    misclass_subs = cell(numiters,task_idx_max);
    correct_subs = cell(numiters,task_idx_max);
    binary_subAccMat = cell(task_idx_max,1);
    
    % phenotype high/low cutoffs from standard scores
    cutoff_low = stdScore_mean(phen_idx)-((1/3)*stdScore_sd(phen_idx));
    cutoff_high = stdScore_mean(phen_idx)+((1/3)*stdScore_sd(phen_idx));
    subset_tot = zeros(min([length(find(datamat(:,phen_idx)<=cutoff_low)) length(find(datamat(:,phen_idx)>=cutoff_high))])*2,nsubs,numiters,task_idx_max);
    
    % if we're doing permutation tests, generate indices to permute only
    % high/low scores
    if permtest==1
        datamat_real = datamat;
        clear datamat;
        np_highlow{phen_idx} = sort([find(datamat_real(:,phen_idx)<=cutoff_low);find(datamat_real(:,phen_idx)>=cutoff_high)],'ascend');
        for iter = 1:numiters
            idx_perm{phen_idx}(:,iter) = randperm(length(np_highlow{phen_idx}));
            while (iter~=1 && any(sum(abs(idx_perm{phen_idx}(:,1:iter-1)-idx_perm{phen_idx}(:,iter)))==0)) || isequal(idx_perm{phen_idx}(:,iter),[1:length(np_highlow{phen_idx})]') % if any of the columns are repeated or permuted indices = true indices, regenerate permuted order
                idx_perm{phen_idx}(:,iter) = randperm(length(np_highlow{phen_idx}));
            end  
        end    
    else
    end
    
    % initialize consensus edges across folds+iterations for given phenotype and task
    edges_pos_tot = zeros(size(mats_toUse,1),task_idx_max);
    edges_neg_tot = zeros(size(mats_toUse,1),task_idx_max);
    
    for task_idx = 1:task_idx_max
        binary_subAccMat{task_idx} = zeros(nsubs,numiters);
        for iter = 1:numiters
            if permtest==1
                clear datamat
                datamat = datamat_real; datamat(np_highlow{phen_idx},phen_idx) = datamat_real(np_highlow{phen_idx}(idx_perm{phen_idx}(:,iter)),phen_idx);
            else
            end
            
            for sub = 1:nsubs % num of total subs
                disp(['working on subject ' num2str(sub) '/' num2str(nsubs) ', task ' num2str(task_idx) '/' num2str(task_idx_max) ', phenotype ' num2str(phen_idx) '/' num2str(phen_idx_max) ', iter ' num2str(iter)]);
                clearvars -except pct_missing savedir phen_idx task_idx iter sub mats_toUse datamat* sublist* names stdScore_* permtest numiters exclude_outliers seed use_tasks *_idx_max nsubs dataset np_highlow idx_perm misclass_subs correct_subs binary_subAccMat cutoff_* edges_pos_tot edges_neg_tot len_train true_label pred_mem* subset_tot
                
                % find high/low training subjects for given phenotype
                score = datamat(:,phen_idx);
                score(sub) = []; % drop test subject
                test_score = (datamat(sub,phen_idx));
                
                % find low and high scorers
                lowScore_dsd_idx = find(score<=cutoff_low); 
                highScore_dsd_idx = find(score>=cutoff_high);
                
                % which is bigger class?
                disp(['length pre-sampling low score = ' num2str(length(lowScore_dsd_idx))]);
                disp(['length pre-sampling high score = ' num2str(length(highScore_dsd_idx))]);
                
                % undersample larger class to be size of smaller class
                num_subs_subset = min([length(lowScore_dsd_idx) length(highScore_dsd_idx)]);
                if length(lowScore_dsd_idx)>length(highScore_dsd_idx)
                    lowScore_dsd_idx = lowScore_dsd_idx(randperm(length(lowScore_dsd_idx),num_subs_subset));
                elseif length(lowScore_dsd_idx)<length(highScore_dsd_idx)
                    highScore_dsd_idx = highScore_dsd_idx(randperm(length(highScore_dsd_idx),num_subs_subset));
                end
                % sanity check that classes are now the same size
                disp(['length post-sampling low score = ' num2str(length(lowScore_dsd_idx))]);
                disp(['length post-sampling high score = ' num2str(length(highScore_dsd_idx))]);
                
                subset_dsd_idx = sort([lowScore_dsd_idx; highScore_dsd_idx],'ascend'); % subset of training subs (highest and lowest) to use for prediction
                len_train(((iter-1)*nsubs)+sub,task_idx) = length(subset_dsd_idx); % matrix of size of training data with rows = nsub folds stacked iter 1, 2, etc, with task = columns
                
                % derive label for test subject
                if test_score<=cutoff_low
                    true_label(sub,iter,task_idx) = -1;
                elseif test_score>=cutoff_high
                    true_label(sub,iter,task_idx) = 1;
                else % no prediction or selected edges for subjects with intermediate or missing scores
                    true_label(sub,iter,task_idx) = 0;
                    disp('test subject not in top or bottom; skipping this iter');
                    pred_mem_svm(sub,iter,task_idx) = NaN;
                    continue
                end
                
                % binary phenotype to classify for training subset of subjects in dsd
                binarized_phen = zeros(size(score,1),1);
                binarized_phen(lowScore_dsd_idx) = -1;
                binarized_phen(highScore_dsd_idx) = 1;
                binarized_phen(binarized_phen==0) = []; % drop intermediate scorers
                train_phen = binarized_phen;
                
                % prepare FC training data
                train_mats_tmp = squeeze(mats_toUse(:,:,task_idx));
                train_mats_tmp(:,sub) = []; % drop test subject
                train_mats = train_mats_tmp(:,subset_dsd_idx); % select training subset
                
                % select correlated edges
                [r,p] = corr(train_mats',train_phen);
                edges_pos = (p < 0.05) & (r > 0);
                edges_neg = (p < 0.05) & (r < 0);
                disp(['num pos edges = ' num2str(length(find(edges_pos))) '; num neg edges = ' num2str(length(find(edges_neg)))]);
                
                % derive edge summary scores for positively and negatively
                % correlated edges in test and train subjects
                test_edgeScore = [sum(squeeze(mats_toUse(edges_pos, sub, task_idx))) sum(squeeze(mats_toUse(edges_neg, sub, task_idx)))];
                for subtrain = 1:size(train_mats,2)
                    train_edgeScore(subtrain,:) = [sum(train_mats(edges_pos, subtrain)) sum(train_mats(edges_neg, subtrain))];
                end
                
                % train SVM
                trainmodel = fitcsvm(zscore(train_edgeScore),train_phen,'KernelFunction','linear'); % classes balanced by undersampling, so no need to set priors and cost
                
                % apply SVM to left-out subject to get predicted score,
                % "z-scoring" test data using training mean and s.d.
                pred_mem_svm(sub,iter,task_idx) = predict(trainmodel,(test_edgeScore-mean(train_edgeScore))./std(train_edgeScore));
                
                % store sets of misclassified and correctly classified subjects
                if (true_label(sub,iter,task_idx)==1 && pred_mem_svm(sub,iter,task_idx)==-1) || (true_label(sub,iter,task_idx)==-1 && pred_mem_svm(sub,iter,task_idx)==1)
                    misclass_subs{iter,task_idx} = [misclass_subs{iter,task_idx}; sub];
                elseif (true_label(sub,iter,task_idx)==1 && pred_mem_svm(sub,iter,task_idx)==1) || (true_label(sub,iter,task_idx)==-1 && pred_mem_svm(sub,iter,task_idx)==-1)
                    correct_subs{iter,task_idx} = [correct_subs{iter,task_idx}; sub];
                else disp('issue assigning misclass and correct subs');
                end
                
                % edges selected across all folds and iterations
                edges_pos_tot(:,task_idx) = edges_pos_tot(:,task_idx)+double(edges_pos);
                edges_neg_tot(:,task_idx) = edges_neg_tot(:,task_idx)+double(edges_neg);
                
                % store subject subsets
                subset_tot(1:length(subset_dsd_idx),sub,iter,task_idx) = subset_dsd_idx;
                subset_tot(find(subset_tot(:,sub,iter,task_idx)>=sub),sub,iter,task_idx) = subset_tot(find(subset_tot(:,sub,iter,task_idx)>=sub),sub,iter,task_idx)+1; % account for dropped subject by increasing each idx after dropped sub by 1
            end
            
            % subject x iteration binary result matrix
            binary_subAccMat{task_idx}(misclass_subs{iter,task_idx},iter) = 1; % misclassified subjects coded as 1; correctly classified = 0;
            binary_subAccMat{task_idx}(find(isnan(squeeze(pred_mem_svm(:,iter,task_idx)))),iter) = NaN; % subjects not classified (intermediate or missing scores) coded as NaN
        end
    end
    % save results separately for each phenotypic measure
    save(strjoin([savedir dataset '/LOOCVclassification_' dataset '_' use_tasks 'FC_' names{phen_idx} '_standardScores_' num2str(numiters) 'iters_p0.05thresh_permTest' string(logical(permtest)) '_outlierExclusion' string(logical(exclude_outliers)) '_excludeSubsMissingOver' num2str(pct_missing) 'pctNode.mat'],''), '-regexp', '^(?!(mats_toUse|train_mats|test_mats)$).'); % save all variables except big matrices
end