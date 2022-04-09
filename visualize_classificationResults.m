% Code to visualize output of CVclassification_standardizedScores_final.m
% July 14, 2021
% Abigail Greene

% initialize some things
homedir = '/data13/mri_group/abby_data/misclassificationAnalyses/results/';
dataset = 'dsd'; % 'dsd' = Yale, vs. 'ucla'
if strcmp(dataset,'ucla')
    tmp = dir(['/data13/mri_group/abby_data/misclassificationAnalyses/data/' upper(dataset) '/ALL*.mat']);
    load([tmp.folder '/' tmp.name],'tasks');
    np = {'ln_std','vocab_std','mr_std'};
else
    load(['/data13/mri_group/abby_data/misclassificationAnalyses/data/' upper(dataset) '/normedNeuropsych_dsd.mat'],'names');
    np = names;
    tasks = {'r1', 'card', 'eyes', 'grad', 'movies', 'nback', 'sst', 'r2'};
end
task_names = tasks; task_names{end+1} = 'gfc';
clear tmp tasks

for i = 1:length(np) % loop through predicted phenotypes
    clear real perm
    real = load([homedir dataset '/LOOCVclassification_' dataset '_bothFC_' np{i} '_standardScores_100iters_p0.05thresh_permTestfalse_outlierExclusiontrue.mat'],'binary_subAccMat','correct_subs','misclass_subs','true_label');
    perm = load([homedir dataset '/LOOCVclassification_' dataset '_bothFC_' np{i} '_standardScores_100iters_p0.05thresh_permTesttrue_outlierExclusiontrue.mat'],'binary_subAccMat','correct_subs','misclass_subs','true_label');
    for j = 1:length(task_names) % loop through in-scanner tasks
        % calculate misclassification frequency (MF)
        misclass_freq_real(:,j,i) = mean(real.binary_subAccMat{j},2);
        misclass_freq_perm(:,j,i) = mean(perm.binary_subAccMat{j},2);
        
        % calculate and plot real vs permuted classification accuracy
        for iter = 1:100
            % calculate overall accuracy
            acc{i}(iter,j) = length(real.correct_subs{iter,j})/length(find(~isnan(real.binary_subAccMat{j}(:,iter))));
            acc_perm{i}(iter,j) = length(perm.correct_subs{iter,j})/length(find(~isnan(perm.binary_subAccMat{j}(:,iter))));
            
            % calculate accuracy separately for high and low scoring subjects
            acc_high{i}(iter,j) = length(intersect(real.correct_subs{iter,j},find(real.true_label(:,iter,j)==1)))/length(find(real.true_label(:,iter,j)==1));
            acc_high_perm{i}(iter,j) = length(intersect(perm.correct_subs{iter,j},find(perm.true_label(:,iter,j)==1)))/length(find(perm.true_label(:,iter,j)==1));
            acc_low{i}(iter,j) = length(intersect(real.correct_subs{iter,j},find(real.true_label(:,iter,j)==-1)))/length(find(real.true_label(:,iter,j)==-1));
            acc_low_perm{i}(iter,j) = length(intersect(perm.correct_subs{iter,j},find(perm.true_label(:,iter,j)==-1)))/length(find(perm.true_label(:,iter,j)==-1));
        end        
    end
    
    % plot accuracy for low and high scorers for the best in-scanner task
    [~,best_task(i)] = max(mean(acc{i}));
    figure(1); subplot(2,ceil(length(np)/2),i); histogram(acc_low{i}(:,best_task(i)),'FaceColor',[0, 109, 119]./255,'FaceAlpha',0.8,'BinWidth',0.05,'Normalization','count'); hold on; histogram(acc_high{i}(:,best_task(i)),'FaceColor',[226, 149, 120]./255,'FaceAlpha',0.8,'BinWidth',0.05,'Normalization','count'); histogram(acc_low_perm{i}(:,best_task(i)),'FaceColor',[131, 197, 190]./255,'FaceAlpha',0.8,'BinWidth',0.05,'Normalization','count'); histogram(acc_high_perm{i}(:,best_task(i)),'FaceColor',[255, 221, 210]./255,'FaceAlpha',0.8,'BinWidth',0.05,'Normalization','count'); xlim([0 1]); hold off; title(['Acc., ' np{i} ', best task: ' task_names{best_task(i)}]);
    
    % calculate significance for best task performance for given NP measure
    p_tot(i) = (length(find(acc_perm{i}(:,best_task(i))>=median(acc{i}(:,best_task(i)))))+1)/(size(acc_perm{i}(:,best_task(i)),1)+1); % uncorrected p values
    
    % visualize misclassification frequencies for all tasks for given np
    % measure, real (red) vs. permuted (blue), all tasks + GFC
    figure(2); subplot(2,ceil(length(np)/2),i); histogram(squeeze(misclass_freq_real(:,:,i)),'FaceColor','r','FaceAlpha',0.5,'BinWidth',0.1); hold on; histogram(squeeze(misclass_freq_perm(:,:,i)),'FaceColor','b','FaceAlpha',0.5,'BinWidth',0.1); hold off; title(['Mean misclass freq, real vs. perm, ' np{i}]);
    
    % are highly misclassified subjects similar within phenotype? should be
    % yes for real, no for permuted (ie low corr)
    corrmat_withinPhen{i} = corr(misclass_freq_real(:,:,i),'rows','pairwise','type','Spearman'); % task x task correlation mat
    corrmat_withinPhen_perm{i} = corr(misclass_freq_perm(:,:,i),'rows','pairwise','type','Spearman');
    % plot correlations
    C = [0 0 1; 1 1 1; 1 0 0]; % blue to red colormap
    % plot real
    figure(3); subplot(4,ceil(length(np)/2),i); imagesc(corrmat_withinPhen{i}); caxis([-1 1]); colormap(interp1(linspace(0,1,length(C)),C,linspace(0,1,250))); xticks(1:length(task_names)); xticklabels(task_names); xtickangle(45); yticks(1:length(task_names)); yticklabels(task_names); ytickangle(45); colorbar; title(['Real data, ' np{i}]);
    % plot permuted underneath
    figure(3); subplot(4,ceil(length(np)/2),i+length(np)); imagesc(corrmat_withinPhen_perm{i}); caxis([-1 1]); colormap(interp1(linspace(0,1,length(C)),C,linspace(0,1,250)));  xticks(1:length(task_names)); xticklabels(task_names); xtickangle(45); yticks(1:length(task_names)); yticklabels(task_names); ytickangle(45); colorbar; title(['Permuted data, ' np{i}]);
    %are correlations significantly higher for real than permuted?
    low_idx_taskmat = find(tril(ones(length(task_names),length(task_names)),-1));
    [pPaired_corrmat_withinPhen(i),~,statsPaired_corrmat_withinPhen{i}] = signrank(corrmat_withinPhen{i}(low_idx_taskmat),corrmat_withinPhen_perm{i}(low_idx_taskmat),'tail','right');
end

% fdr correct accuracy p vals
p_tot_fdr = mafdr(p_tot,'BHFDR',1);

% correlate misclassification frequency for given task across phenotypes,
% then average across tasks
for j = 1:size(misclass_freq_real,2)
    corrmat_acrossPhen(:,:,j) = corr(squeeze(misclass_freq_real(:,j,:)),'rows','pairwise','type','Spearman');
    corrmat_acrossPhen_perm(:,:,j) = corr(squeeze(misclass_freq_perm(:,j,:)),'rows','pairwise','type','Spearman');
    % find number of overlapping subjects for each correlation
    ct_corr=1;
    for i = 1:length(np)
        tmp_idx = 1:length(np); tmp_idx(i) = [];
        for ii = tmp_idx
            num_corr_scores(ct_corr,j) = length(intersect(find(~isnan(misclass_freq_real(:,j,i))),find(~isnan(misclass_freq_real(:,j,ii)))));
            num_corr_scores_perm(ct_corr,j) = length(intersect(find(~isnan(misclass_freq_perm(:,j,i))),find(~isnan(misclass_freq_perm(:,j,ii)))));
            ct_corr=ct_corr+1;
        end
    end
    disp([task_names{j} ': min #subs correlated across phenotypes = ' num2str(min(num_corr_scores(:,j))) ', max = ' num2str(max(num_corr_scores(:,j)))]);
    disp(['perm: ' task_names{j} ': min #subs correlated across phenotypes = ' num2str(min(num_corr_scores_perm(:,j))) ', max = ' num2str(max(num_corr_scores_perm(:,j)))]);   
end
% average matrices across correlations to get overall measure-by-measure
% correlation matrix
corrmat_acrossPhen_avg = mean(corrmat_acrossPhen,3); % np x np correlation matrix
corrmat_acrossPhen_perm_avg = mean(corrmat_acrossPhen_perm,3);

% how similar (ie correlated) are phenotypic scores, themselves?
load([homedir dataset '/LOOCVclassification_' dataset '_bothFC_' np{i} '_standardScores_100iters_p0.05thresh_permTestfalse_outlierExclusiontrue.mat'],'datamat'); % use np scores from which outliers (3+sd below the mean) have already been excluded (ie converted to NaN) during CVclassification_standardizedScores_final.m
corrmat_phen = corr(datamat,'rows','pairwise','type','Spearman');

% compare measure and MF similarity
% first, scatterplot (left: real, right: permuted)
low_idx = find(tril(ones(size(corrmat_phen,1),size(corrmat_phen,2)),-1));
figure(4); subplot(1,2,1); scatter(corrmat_phen(low_idx),corrmat_acrossPhen_avg(low_idx),75,'k','filled'); lsline; xlabel('Measure correlation'); ylabel('Misclassification frequency correlation'); title('Real data'); hold on; an1=annotation('textbox',[0.1 0.1 0.4 0.1],'String',['r_s = ' num2str(corr(corrmat_phen(low_idx),corrmat_acrossPhen_avg(low_idx),'type','Spearman'))],'FontSize',16,'EdgeColor','none'); hold off;
figure(4); subplot(1,2,2); scatter(corrmat_phen(low_idx),corrmat_acrossPhen_perm_avg(low_idx),75,'k','filled'); lsline; xlabel('Measure correlation'); ylabel('Misclassification frequency correlation'); title('Permuted data'); hold on; an2=annotation('textbox',[0.7 0.1 0.4 0.1],'String',['r_s = ' num2str(corr(corrmat_phen(low_idx),corrmat_acrossPhen_perm_avg(low_idx),'type','Spearman'))],'FontSize',16,'EdgeColor','none'); hold off;
% now, via hierarchical linkage, visualized in dendrogram
z = linkage(1-corrmat_acrossPhen_avg(low_idx)');
figure(5); dendrogram(z,0,'labels',np); title('Misclassification frequency similarity across phenotypic measures');