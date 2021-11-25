clear;
list_filename='trainingfile_list.mat';
MODEL_NO=11;
dir1='mfcc';
% dir3={'AE','AJ','AL','AW','BD','CB','CF','CR','DL','DN','EH','EL','FC','FD','FF','FI','FJ','FK','FL','GG','GJ','GR','GT','HA','HH','HL','HN','HS','IE','IF','IN','IT','JR','JT','KD','KP','KR'};
dir3={'AE','AJ','AL','AW','BD','CB','CF','CR','DL','DN','EH','EL','FC','FD','FF','FI','FJ','FK','FL','GG'};
wordids={'1','2','3','4','5','6','7','8','9','O','Z'};

k=1;
for d3=1:length(dir3)
    for w=1:length(wordids)
        for pass='A':'B'
            trainingfile{k,1}=w;
            trainingfile{k,2}=sprintf('%s/%s/%c%c_endpt.mfc', dir1,dir3{d3},wordids{w},pass);
            k=k+1;
        end
    end
end
save(list_filename,'trainingfile');
