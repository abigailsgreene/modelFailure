% Code to relate misclassification frequency and mean scorec to covariates
% July 14, 2021
% Abigail Greene

% Set some things
homedir = '/data13/mri_group/abby_data/misclassificationAnalyses/data/';
dataset = 'dsd'; % 'dsd' = Yale vs. 'ucla'
exclude_outliers=1; % 1=yes,0=no

% load in covariates
covar_file = dir([homedir upper(dataset) '/covariates_*.mat']);
load([covar_file.folder '/' covar_file.name]);

% load in neuropsych data
if strcmp(dataset,'ucla')
    [~,~,datamat_tmp] = xlsread([homedir upper(dataset) '/normedNeuropsych.xlsx']); % load in three scores (WAIS LN, vocab, and MR, normed by AG)
    names = {'ln_std', 'vocab_std','mr_std'};
    stdScore_mean = [10 10 10]; stdScore_sd = [3 3 3]; % WAIS = scaled scores
    for ucla_np = 1:length(names)
        col_idx = find(strcmp(datamat_tmp(1,:),names{ucla_np}));
        datamat(:,ucla_np) = cell2mat(datamat_tmp(2:end,col_idx));
    end
    np_labels = {'ln','vocab','mr'};
elseif strcmp(dataset,'dsd')
    load([homedir upper(dataset) '/normedNeuropsych_dsd.mat']); % names = measures, in order; datamat = subject x measure matrix of normed scores
    % set norms
    for i = 1:length(names)
        if contains(names{i},'wrat') % IQ score (mean 100, std 15)
            stdScore_mean(i) = 100; stdScore_sd(i) = 15;
        elseif contains(names{i},{'brief','wasi'}) % t scores (mean 50, std 10)
            stdScore_mean(i) = 50; stdScore_sd(i) = 9; % see rounding note in CVclassification_standardizedScores_final.m
        elseif contains(names{i},{'wraml','wais','dkefs'}) % scaled scores (mean 10, std 3)
            stdScore_mean(i) = 10; stdScore_sd(i) = 3;
        elseif contains(names{i},{'bnt'}) % z scores
            stdScore_mean(i) = 0; stdScore_sd(i) = 1;
        end
    end
    np_labels = {'bnt','wrat','vl','vl,delay','fw','symbol','coding','LN','cancellation','trails','VF1','VF2','CW','20Q','vocab','mr'};
end

% exclude subjects with outlier (>3sd <mean) phenotypic data based on population norms (i.e., code their data as nan)
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

% run pca on z-scored phenotypic data
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca((datamat-nanmean(datamat))./nanstd(datamat));

% is misclassification frequency related to diagnosis for ucla?
if strcmp(dataset,'ucla')
    load([homedir upper(dataset) '/scid_163subs.mat']);
    hc_idx = []; pt_idx = [];
    for i = 1:size(scid163,1)
        if contains(scid163.dx1(i,:),'No Diagnosis') || contains(scid163.dx1(i,:),'n/a')
            hc_idx = [hc_idx; i];
        else pt_idx = [pt_idx; i];
        end
    end
    pt_hc_bin = zeros(size(scid163,1),1); pt_hc_bin(pt_idx) = 1;
end

% load motion data
motion_file = dir([homedir upper(dataset) '/*motion*.mat']);
load([motion_file.folder '/' motion_file.name]);
if strcmp(dataset,'dsd')
    motion_all=f2f;
end

% aggregate covariates and relate them to misclass freq
if strcmp(dataset,'ucla')
    covar_all = [sex_tot age_tot race_tot hopkins_tot schoolyrs_tot medStatus_tot pt_hc_bin SCORE(:,1:3) mean(motion_all,2)];
    covar_names = {'sex','age','race','sx','edu','rx','dx','PCscore1','PCscore2','PCscore3','motion'};
elseif strcmp(dataset,'dsd')
    covar_all = [datamat_sex datamat_age datamat_race datamat_gsi datamat_edu datamat_pss datamat_psqi datamat_rx datamat_dx SCORE(:,1:3) mean(motion_all,2)];
    covar_names = {'sex' 'age' 'race' 'gsi' 'edu' 'pss' 'psqi' 'rx' 'dx' 'PCscore1' 'PCscore2' 'PCscore3' 'motion'};
else
end

% calculate mean MF per neuropsych measure (using misclass_freq_real from
% visualize_classificationResults.m)
misclass_freq_perNP = squeeze(mean(misclass_freq_real,2));

% relate covariates to single MF per subject, separately averaging MFs from measures on which they scored high (misclass_freq_overall_high) and on which they scored low (misclass_freq_overall_low)
for i = 1:length(names)
    load(['/data13/mri_group/abby_data/misclassificationAnalyses/results/' dataset '/LOOCVclassification_' dataset '_bothFC_' names{i} '_standardScores_100iters_p0.05thresh_permTestfalse_outlierExclusiontrue.mat'],'true_label');
    if sum(var(reshape(true_label,size(true_label,1),size(true_label,2)*size(true_label,3)),[],2))~=0
        disp('uhoh, labels differ across iters')
        break
    else
        true_label_tot(:,i) = squeeze(true_label(:,1,1)); % label should be the same across tasks and iters
    end
    clear true_label
end

misclass_freq_perNP_low = nan(size(true_label_tot));
misclass_freq_perNP_high = nan(size(true_label_tot));
misclass_freq_perNP_low(find(true_label_tot==-1)) = misclass_freq_perNP(find(true_label_tot==-1)); % subject x measure matrix of MF for low scores only
misclass_freq_perNP_high(find(true_label_tot==1)) = misclass_freq_perNP(find(true_label_tot==1)); % subject x measure matrix of MF for high scores only
misclass_freq_overall_high = nanmean(misclass_freq_perNP_high,2);
misclass_freq_overall_low = nanmean(misclass_freq_perNP_low,2);

for i = 1:size(covar_all,2)
    if isempty(find(~ismember(covar_all(~isnan(covar_all(:,i)),i),[0 1]))) % binary covariates, so do two-sided ranksum here
        [p_high(i),~,stats_high{i}] = ranksum(misclass_freq_overall_high(find(covar_all(:,i)==0)),misclass_freq_overall_high(find(covar_all(:,i)==1)));
        [p_low(i),~,stats_low{i}] = ranksum(misclass_freq_overall_low(find(covar_all(:,i)==0)),misclass_freq_overall_low(find(covar_all(:,i)==1)));
        ranksum_low(i) = stats_low{i}.ranksum;
        ranksum_high(i) = stats_high{i}.ranksum;
        medianDiff_mf_high(i) = nanmedian(misclass_freq_overall_high(find(covar_all(:,i)==0)))-nanmedian(misclass_freq_overall_high(find(covar_all(:,i)==1)));
        medianDiff_mf_low(i) = nanmedian(misclass_freq_overall_low(find(covar_all(:,i)==0)))-nanmedian(misclass_freq_overall_low(find(covar_all(:,i)==1)));
    else
        [r_high(i),p_high(i)] = corr(misclass_freq_overall_high,covar_all(:,i),'type','Spearman','rows','pairwise');
        [r_low(i),p_low(i)] = corr(misclass_freq_overall_low,covar_all(:,i),'type','Spearman','rows','pairwise');
    end
end
% FDR correct p values
p_tot_fdr = mafdr([p_low(:);p_high(:)],'BHFDR',1);

% now relate [significant] variables to mean score (aftr z-scoring within
% measure)
datamat_z = (datamat-nanmean(datamat))./nanstd(datamat); 

% separate scores when subjects were correctly and incorrectly classified
datamat_z_correct = nan(size(misclass_freq_perNP,1),size(misclass_freq_perNP,2));
datamat_z_misclass = nan(size(misclass_freq_perNP,1),size(misclass_freq_perNP,2));
for i = 1:size(misclass_freq_perNP,2) % loop through measures
    % find correctly and incorrectly classified subjects for given measure
    idx_misclass = find(misclass_freq_perNP(:,i)>=0.5); idx_correct = find(misclass_freq_perNP(:,i)<0.5);
    datamat_z_correct(idx_correct,i) = datamat_z(idx_correct,i);
    datamat_z_misclass(idx_misclass,i) = datamat_z(idx_misclass,i);
    clear idx*
end

for i = 1:size(covar_all,2)
    if isempty(find(~ismember(covar_all(~isnan(covar_all(:,i)),i),[0 1]))) % binary covariates, so do two-sided ranksum here
        [p(i,1),~,score_stats_correct{i}] = ranksum(nanmean(datamat_z_correct(find(covar_all(:,i)==0),:),2),nanmean(datamat_z_correct(find(covar_all(:,i)==1),:),2));
        [p(i,2),~,score_stats_misclass{i}] = ranksum(nanmean(datamat_z_misclass(find(covar_all(:,i)==0),:),2),nanmean(datamat_z_misclass(find(covar_all(:,i)==1),:),2));
        ranksum_score_correct(i) = score_stats_correct{i}.ranksum;
        ranksum_score_misclass(i) = score_stats_misclass{i}.ranksum;
        medianDiff_score_correct(i) = nanmedian(nanmean(datamat_z_correct(find(covar_all(:,i)==0),:),2))-nanmedian(nanmean(datamat_z_correct(find(covar_all(:,i)==1),:),2));
        medianDiff_score_misclass(i) = nanmedian(nanmean(datamat_z_misclass(find(covar_all(:,i)==0),:),2))-nanmedian(nanmean(datamat_z_misclass(find(covar_all(:,i)==1),:),2));
    else
        [r(i,1),p(i,1)] = corr(covar_all(:,i),nanmean(datamat_z_correct,2),'type','Spearman','rows','pairwise');
        [r(i,2),p(i,2)] = corr(covar_all(:,i),nanmean(datamat_z_misclass,2),'type','Spearman','rows','pairwise');
    end
    
end