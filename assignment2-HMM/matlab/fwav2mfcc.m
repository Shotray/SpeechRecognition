function feature_seq=fwav2mfcc(infilename,outfilename,outfile_format,frame_size_sec,frame_shift_sec,use_hamming,pre_emp,bank_no,cep_order,lifter,delta_win_weight)

[audioIn, fs]=audioread(infilename);
len=length(audioIn);
%pre-emphasis
speech=zeros(len,1);
speech(1)=audioIn(1);
speech(2:end)=audioIn(2:end)-pre_emp*audioIn(1:end-1);

winlen = fs*0.025; % window length = 200 points, ie 25ms
overlap = 120;  % overlap: 120 points
[coeffs,delta,deltaDelta,loc] = mfcc(speech, fs, 'WindowLength', winlen, 'OverlapLength', overlap, 'NumCoeffs', 12);
feature_seq = [coeffs'; delta'; deltaDelta'];
[dim frame_no]=size(feature_seq);

switch lower(outfile_format)
    case 'htk' % write htk header, big endian
        fout=fopen(outfilename,'w','b'); % 'n'==local machine format 'b'==big endian 'l'==little endian
        fwrite(fout,frame_no,'int32');
        sampPeriod=round(frame_shift_sec*1E7);    
        fwrite(fout,sampPeriod,'int32');
        sampSize=dim*4;   
        fwrite(fout,sampSize,'int16');
        parmKind=838; % parameter kind code: MFCC=6, _E=64, _D=256, _A=512, MFCC_E_D_A=6+64+256+512=838
        fwrite(fout,parmKind,'int16'); 
end
fwrite(fout, feature_seq,'float32');
fclose(fout);

end