clear

fs = 16000;
pow_thresh=-20;                           % To test recording failure

addpath ../utils;
upath='../../data/audio/16kHz/isolated/'; % path to segmented utterances
cpath='../../data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../../data/audio/16kHz/backgrounds/'; % path to noise backgrounds
apath='../../data/annotations/'; % path to JSON annotations

datapath = '/scratch/mensa/xiali/CHiME3/data/audio/16kHz/NBDF/';

sets = {'tr','dt','et'};
ori_spksets = {'M01 M02 F02 F03','M03 M04 F01 F04','M05 F05 M06 F06'};   % CHiME3 original speaker sets for training, development and test, respectively

%% CHiME3 background noise, bth utterances %%

% background nosie
blist = dir([bpath '*.CH1.wav']);

% bth utterances
bth_tr_mat=json2mat([apath 'tr05_bth.json']);
bth_dt_mat=json2mat([apath 'dt05_bth.json']);
bth_et_mat=json2mat([apath 'et05_bth.json']);
bth_mat = [bth_tr_mat,bth_dt_mat,bth_et_mat];

%% Speakers and background noise used for train, validation and test of narrow-band deep filtering network %%

tr_spks = 'M01 M02 F02 F03'; % Speakers used for training of narrow-band deep filtering network
dt_spks = 'M01 M02 F02 F03';
et_spks = 'F01 M03 M04 F04 M05 F05 M06 F06';

tr_nt = 'BUS CAF PED STR'; % Noise types used for training of narrow-band deep filtering network
dt_nt = 'BUS CAF PED STR';
et_nt = 'BUS CAF PED STR';

% Collect utterances into tr_utt, dt_utt and et_utt for train, validation and test, respectively. 
% Similarly, noise sessions are collected into *_noise 
for setindx = 1:length(sets)
    set = sets{setindx};
    eval([set '_utt={};']);
    eval([set '_noise=[];']);
end

for uind = 1:length(bth_mat)
    for setindx = 1:length(sets)
        set = sets{setindx};
        eval(['iscon=contains(' set '_spks,bth_mat{uind}.speaker);']);
        if iscon    
            eval([set '_utt(end+1)= bth_mat(uind);']);
        end
    end    
end

for nind = 1:length(blist)
    nname = blist(nind).name(end-10:end-8);
    for setindx = 1:length(sets)
        set = sets{setindx};
        eval(['iscon=contains(' set '_nt,nname);']);
        if iscon    
            if isempty(eval([set '_noise']))
                eval([set '_noise= blist(nind);']);
            else
                eval([set '_noise(end+1)= blist(nind);']);
            end
        end
    end       
end

%% Generate mixed data
CHAN = [3 4 5 6];
refChan = 6;
refIndx = find(CHAN==refChan);
nchan = length(CHAN);

% output folders
outfolders = {'train_mixed_wav','validation_mixed_wav','test_mixed_wav'}; 

% For data augmentation, each clean utterance is mixed with multiple randomly selected background noise segments
ITE = [15,3,5];              

% Randomly selceted multiple clean utterances are first concatenated, then 
% from which extract training/validation/test sequences
% Utterances are speaked by different speakers and from different
% locations, thence training with concatenated signals takes speech/speaker
% turns into account
UTTCONC = [10,10,1];       

% First 60% noise recording of each session is used for train/validation, and last 40% for test 
Npoint = [0 0.6; 0 0.6;0.6 1];  

% Mixing SNRs
TRAINSNR = [-5,10]; % Train/validation snr range
TESTSNR = -4:4:8;   % Test SNRs

% Generate mixed train/validation/test data
for i = 1:3
    
    set = sets{i};
    ite = ITE(i);
    uttconc = UTTCONC(i);
    eval(['noise_session=',set,'_noise;']);
    bn = length(noise_session);
    
    outpath = [datapath outfolders{i}];
    if ~exist(outpath,'dir')
        system(['mkdir -p ' outpath]);
    end    
    
    for j=1:ite       
        fprintf([set ' data: ' num2str(j) '/' num2str(ite) ' \n'])
        eval(['clean_utt=',set,'_utt;']);
        clean_utt = clean_utt(randperm(length(clean_utt))); % shuffle clean utterances               
        
        %
        for utt_indx = 1:uttconc:length(clean_utt)
            
            % Clean utterances concatenation
            s =  [];
            for utt_ind = utt_indx:min(utt_indx+uttconc-1,length(clean_utt))
                ori_spkset =  find(contains(ori_spksets,clean_utt{utt_ind}.speaker));
                               
                oname=[clean_utt{utt_ind}.speaker '_' clean_utt{utt_ind}.wsj_name '_BTH'];                
                su = [];
                c = 0;
                for ch = CHAN
                    c = c+1;
                    su(:,c) =  audioread([upath sets{ori_spkset} '05_bth/' oname '.CH' num2str(ch) '.wav']);
                end
                spow=sum(su.^2,1);
                spow=10*log10(spow/max(spow));
                sfail=any(spow<=pow_thresh);
                if sfail
                    continue;
                end               
                
                s = [s;su];
            end
            sdur = length(s);
            if sdur == 0
                continue;
            end
            
            % Extract background noise 
            nfail = true;
            while nfail
                bname=noise_session(randperm(bn,1)).name(1:end-8); % draw a random background recording
                bb=audioinfo([bpath bname '.CH1.wav']);
                bdur=bb.Duration*fs;
                
                setbeg = round(Npoint(i,1)*bdur);
                setend = round(Npoint(i,2)*bdur)-sdur;
                
                nbeg = round(rand(1)*(setend-setbeg))+1;
                nbeg = setbeg+nbeg;
                nend = nbeg+sdur-1;
                n=zeros(sdur,nchan);
                c = 0;
                for ch=CHAN
                    c = c+1;
                    n(:,c)=audioread([bpath bname '.CH' int2str(ch) '.wav'],[nbeg nend]);
                end
                npow=sum(n.^2,1);
                npow=10*log10(npow/max(npow));
                nfail=any(npow<=pow_thresh);
            end            
            envir = lower(bname(end-2:end));
            
            if i <= 2                % training and validation data
                snrdb = rand(1)*(TRAINSNR(2)-TRAINSNR(1))+TRAINSNR(1);            
                snr = 10^(snrdb/10);                
                s=sqrt(snr/sum(sum(s.^2))*sum(sum(n.^2)))*s;
                x=s+n;                                
                x(:,end+1) = s(:,refIndx);              % multichannel noisy signal and reference channel clean speech           
                audiowrite([outpath '/' envir '_' num2str(snrdb) 'dB.wav'],x/max(max(abs(x))),fs);                  
            else                     % test data
                snrdb = TESTSNR(randperm(length(TESTSNR),1));
                snr = 10^(snrdb/10);                
                s=sqrt(snr/sum(sum(s.^2))*sum(sum(n.^2)))*s;
                x=s+n;    
                                
                udir=[outpath '/'  envir '/' 'snr' num2str(snrdb) '/'];
                if ~exist(udir,'dir')
                    system(['mkdir -p ' udir]);
                end                
                                      
                audiowrite([udir oname(1:end-4) '_ms.wav'],x/max(max(abs(x))),fs);   
                audiowrite([udir oname(1:end-4) '_refms.wav'],x(:,refIndx)/max(abs(x(:,refIndx))),fs);   
                audiowrite([udir oname(1:end-4) '_cln.wav'],s(:,refIndx)/max(abs(s(:,refIndx))),fs); 
            end
        end
    end
end







