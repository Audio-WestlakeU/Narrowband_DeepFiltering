clear 

fs = 16000;

datapath = '/scratch/mensa/xiali/CHiME3/data/audio/16kHz/NBDF/';

testdata = 'test_mixed_wav';
enhanced = {'blstm-mrm-3456chs-enhanced','blstm-cirm-3456chs-enhanced','blstm-cc-3456chs-enhanced','blstm-sf-3456chs-enhanced'};

SNR = -4:4:8;
envirs = {'bus','caf','ped','str'};

Srmr = zeros(length(envirs),length(SNR),length(enhanced)+1);
PESQ = zeros(length(envirs),length(SNR),length(enhanced)+1);
SDR = zeros(length(envirs),length(SNR),length(enhanced)+1);
STOI = zeros(length(envirs),length(SNR),length(enhanced)+1);

for envind = 1:length(envirs)
    env = envirs{envind};
    for snrind = 1:length(SNR)
        snr = SNR(snrind);               
        mixedpath = [datapath testdata '/' env '/snr' num2str(snr) '/'];        
        files = dir([mixedpath '*_ms.wav']);
        nfiles = length(files);
        
        fprintf(['Processing ' env ' / ' num2str(snr) ' dB / ' num2str(nfiles) ' utterences ... \n'])        
        for i = 1:nfiles
            
            ms_name = files(i).name;
            cln_name = [ms_name(1:12) '_cln.wav'];
            refms_name = [ms_name(1:12) '_refms.wav'];
            
            spath = [mixedpath,cln_name];
            ypath = [mixedpath,refms_name];
            
            s = audioread(spath); s = s(:);
            y = audioread(ypath); y = y(:);
            
            % scores of unprocessed siganl
            sl = min(length(y),length(s));
            pesqy = pesq('+16000',spath,ypath);
            [sdry,~,~] = bsseval(y(1:sl)',s(1:sl)',512);
            stoiy = taal2011(s(1:sl), y(1:sl), fs);
            srmry = SRMR(y,fs, 'fast', 1, 'norm', 1);
            
            Srmr(envind,snrind,end) = Srmr(envind,snrind,end)+ srmry;
            PESQ(envind,snrind,end) = PESQ(envind,snrind,end)+pesqy;
            SDR(envind,snrind,end) = SDR(envind,snrind,end)+sdry;
            STOI(envind,snrind,end) = STOI(envind,snrind,end)+stoiy;
            
            % scores of enhanced
            for enhind = 1:length(enhanced)
                xpath =  [datapath enhanced{enhind} '/'  env '/snr' num2str(snr) '/' ms_name(1:12) '.wav'];
                x = audioread(xpath); x = x(:);
                audiowrite(xpath,x,fs);
                                             
                sl = min(length(x),length(s));
                pesqx = pesq('+16000',spath,xpath);
                [sdrx,~,~] = bsseval(x(1:sl)',s(1:sl)',512);
                stoix = taal2011(s(1:sl), x(1:sl), fs);
                srmrx = SRMR(x,fs, 'fast', 1, 'norm', 1);
                
                Srmr(envind,snrind,enhind) = Srmr(envind,snrind,enhind)+srmrx;
                PESQ(envind,snrind,enhind) = PESQ(envind,snrind,enhind)+pesqx;
                SDR(envind,snrind,enhind) = SDR(envind,snrind,enhind)+sdrx;
                STOI(envind,snrind,enhind) = STOI(envind,snrind,enhind)+stoix;                
            end
            
            
        end
        Srmr(envind,snrind,:) = Srmr(envind,snrind,:)/nfiles;
        PESQ(envind,snrind,:) = PESQ(envind,snrind,:)/nfiles;
        SDR(envind,snrind,:) = SDR(envind,snrind,:)/nfiles;
        STOI(envind,snrind,:) = STOI(envind,snrind,:)/nfiles;        
    end
end

respath = 'se_result/';
if ~exist(respath,'dir')
    system(['mkdir -p ' respath]);
end
save([respath 'PESQ.mat'],'PESQ')
save([respath 'SDR.mat'],'SDR')
save([respath 'STOI.mat'],'STOI')
save([respath 'Srmr.mat'],'Srmr')


