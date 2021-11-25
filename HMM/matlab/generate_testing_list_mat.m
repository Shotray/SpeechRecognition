clear;
list_filename='testingfile_list.mat';
MODEL_NO=11;
dir1='mfcc';
dir3={'AH','AR','AT','BC','BE','BM','BN','CC','CE','CP','DF','DJ','ED','EF','ET','FA','FG','FH','FM','FP','FR','FS','FT','GA','GP','GS','GW','HC','HJ','HM','HR','IA','IB','IM','IP','JA','JH','KA','KE','KG','LE','LG','MI','NL','NP','NT','PC','PG','PH','PR','RK','SA','SL','SR','SW','TC'};
wordids={'1','2','3','4','5','6','7','8','9','O','Z'};

k=1;
for d3=1:length(dir3)
    for w=1:length(wordids)
        for pass='A':'B'
            testingfile{k,1}=w;
            testingfile{k,2}=sprintf('%s/%s/%c%c_endpt.mfc', dir1,dir3{d3},wordids{w},pass);
            k=k+1;
        end
    end
end
% save(list_filename,'list');
save(list_filename,'testingfile');