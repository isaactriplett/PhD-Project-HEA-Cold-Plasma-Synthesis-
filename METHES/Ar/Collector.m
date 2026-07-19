% collect_methes_Ar.m — assemble the E/N sweep into a transport table.
resDir = '.\results_Ar';
F = dir(fullfile(resDir,'Ar_EN_*Td.mat'));
n = numel(F);
EN=nan(n,1); Emean=nan(n,1); w=nan(n,1);
DNt=nan(n,1); DNl=nan(n,1); kion=nan(n,1); keff=nan(n,1); katt=nan(n,1);

for j=1:n
    S=load(fullfile(resDir,F(j).name)); r=S.results;
    EN(j)    = double(r.EN);
    Emean(j) = r.E.E_mean;
    w(j)     = abs(r.flux.w(3));        % drift speed along field (m/s)
    DNt(j)   = mean(r.flux.DN(1:2));    % transverse  D*N (1/m/s)
    DNl(j)   = r.flux.DN(3);            % longitudinal D*N (1/m/s)
    kion(j)  = r.rates.conv.ion_tot;    % m^3/s  (EEDF-convolution estimate)
    keff(j)  = r.rates.conv.eff;        % m^3/s
    katt(j)  = r.rates.conv.att_tot;    % m^3/s
end

[EN,ix]=sort(EN); Emean=Emean(ix); w=w(ix); DNt=DNt(ix); DNl=DNl(ix);
kion=kion(ix); keff=keff(ix); katt=katt(ix);

muN    = w ./ (EN*1e-21);   % reduced mobility       1/(m V s)
aN_eff = keff ./ w;         % effective Townsend a/N  (m^2)
aN_ion = kion ./ w;         % ionization a/N          (m^2)

T=table(EN,Emean,w,muN,DNt,DNl,kion,katt,keff,aN_ion,aN_eff, ...
 'VariableNames',{'EN_Td','Emean_eV','w_ms','muN','DNt','DNl', ...
                  'kion_m3s','katt_m3s','keff_m3s','alphaN_ion_m2','alphaN_eff_m2'});
writetable(T,'Ar_swarm_table.csv'); save('Ar_swarm_table.mat','T');
fprintf('Wrote Ar_swarm_table.csv (%d points)\n',height(T));