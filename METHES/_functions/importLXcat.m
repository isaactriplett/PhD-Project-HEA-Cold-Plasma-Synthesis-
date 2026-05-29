%     Copyright (c) 2015, Mohamed Rabie
%     All rights reserved.
%     
%     Redistribution and use in source and binary forms, with or without
%     modification, are permitted provided that the following conditions are
%     met:
%     - Redistributions of source code must retain the above copyright notice, 
%     this list of conditions and the following disclaimer.
%     - Redistributions in binary form must reproduce the above copyright notice, 
%     this list of conditions and the following disclaimer in the documentation 
%     and/or other materials provided with the distribution
%     - Proper reference is made in publications reporting results obtained 
%     using this software. At present, the preferred way to reference METHES is as follows: 
%     M. Rabie and C.M. Franck, "A Monte Carlo collision code for electron transport in low temperature plasmas", 
%     to be submitted to J. Phys. D: Appl. Phys. (2015) 
%     as well as M. Rabie and C.M. Franck, METHES code, downloaded from www.lxcat.net/download/METHES/, 2015
%         
%     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%     ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%     LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%     CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%     SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%     INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%     CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%     ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%     POSSIBILITY OF SUCH DAMAGE.

classdef importLXcat
    
    properties
        
        % cell array of directories of LXcat files
        dir
        % ionization cross section
        ion
        % attachment cross section
        att
        % elastic cross section
        ela
        % excitation cross section
        exc
        % momentum transfer cross section
        mom
        % effective momentum transfer cross section
        eff
        % threshold energies of ionization
        ionThresh
        % threshold energies of exciations
        excThresh
        % total cross secion
        tot
        % vector giving discret energies
        energy
        % plot (interactive=1) or do not plot data (interactive=0)
        interactive
            
    end
    
    methods
        
        
        % =====================================================================
        %> @brief loads all cross sections from LXcat-files
        %> Parameters: [ion,att,ela,exc,ionThresh,excThresh]
        % =====================================================================
        function obj = importXsections(obj)
            % loads all cross sections from LXcat-files 
            % into arrays ion, att, ela, exc, mom and eff 
            % and threshold energies into arrays ionThresh and excThresh
            
            obj = ionization(obj);
            obj = attachment(obj);
            obj = elastic(obj);
            obj = excitation(obj);
            obj = momentum(obj);
            obj = effective(obj);
            
        end
        
        % =====================================================================
        %> @brief loads ionization cross sections from LXcat-files
        %> Parameters: [ion,ionThresh]
        % =====================================================================
        function obj = ionization(obj)
            % loads ionization cross sections from LXcat-files 
            % into array ion and ionization threshold energy into ionThresh
            
            for j = 1 : length(obj.dir)
                
                file = dir([obj.dir{j} '/*.txt']);
                fid = fopen([obj.dir{j} '/' file.name]);
                
                C = textscan(fid, '%s %s %s %s %s %s %s %s' );
                
                x = strfind(C{1,1}, 'IONIZATION');
                K = sum(cell2mat(x));
                
                if K == 0
                    
                    obj.ion{j} = [];
                    
                else
                    
                    
                    next = 1 ;
                    for k = 1 : K
                        
                        sigma = 0;
                        i = next ;
                        while  isempty(x{i})
                            i = i+1;
                        end
                        ind = i;
                        
                        ind_ion = ind + 2;
                        
                        x = strfind(C{1,1}, '---');
                        i = ind;
                        while isempty(x{i})
                            i = i+1;
                        end
                        first = i+1;
                        
                        i = first;
                        while isempty(x{i})
                            i = i+1;
                        end
                        last = i-1;
                        
                        N = length(first:last);
                        for i = 1 : N
                            ind = first+i-1;
                            sigma(i,1) = str2num(cell2mat(C{1,1}(ind)));
                            sigma(i,2) = str2num(cell2mat(C{1,2}(ind)));
                            
                        end
                        
                        obj.ion{j}{k} = sigma;
                        obj.ionThresh{j}{k} = str2num(cell2mat(C{1,1}(ind_ion)));
                        
                        next = last + 3 ;
                    end
                    
                end
                
            end
            
            %%
            
            fclose(fid);
            
        end
        
        % =====================================================================
        %> @brief loads attachment cross sections from LXcat-files
        %> Parameters: [att]
        % =====================================================================
        function obj = attachment(obj)
            % loads attachment cross sections from LXcat-files 
            % into array att 
            
            for j = 1 : length(obj.dir)
                
                file = dir([obj.dir{j} '/*.txt']);
                fid = fopen([obj.dir{j} '/' file.name]);
                
                C = textscan(fid, '%s %s %s %s %s %s %s %s' );
                
                x = strfind(C{1,1}, 'ATTACHMENT');
                xx = strfind(C{1,1}, '---');
                K = sum(cell2mat(x));
                
                if K == 0
                    
                     obj.att{j} = [];
                    
                else
                    
                    
                    next = 1 ;
                    for k = 1 : K
                        
                        sigma = 0;
                        
                        i = next ;
                        while  isempty(x{i})
                            i = i+1;
                        end
                        ind = i;
                        
                        
                        i = ind;
                        while isempty(xx{i})
                            i = i+1;
                        end
                        first = i+1;
                        
                        i = first;
                        while isempty(xx{i})
                            i = i+1;
                        end
                        last = i-1;
                        
                        N = length(first:last);
                        for i = 1 : N
                            ind = first+i-1;
                            sigma(i,1) = str2num(cell2mat(C{1,1}(ind)));
                            sigma(i,2) = str2num(cell2mat(C{1,2}(ind)));
                        end
                        
                        obj.att{j}{k} = sigma;
                        
                        next = last + 1 ;
                    end
                    
                end
                
            end
            %%
            
            fclose(fid);
            
        end
        
        % =====================================================================
        %> @brief loads elastic cross sections from LXcat-files
        %> Parameters: [ela]
        % =====================================================================
        function obj = elastic(obj)
            % loads elastic cross sections from LXcat-files 
            % into array ela
            
            for j = 1 : length(obj.dir)
                
                file = dir([obj.dir{j} '/*.txt']);
                fid = fopen([obj.dir{j} '/' file.name]);
                
                C = textscan(fid, '%s %s %s %s %s %s %s %s' );
                
                x = strfind(C{1,1}, 'ELASTIC');
                K = sum(cell2mat(x));
                
                if K == 0
                    
                    obj.ela{j} = [];
                    
                else
                    
                    
                    next = 1 ;
                    for k = 1 : K
                        sigma = 0;
                        i = next ;
                        while  isempty(x{i})
                            i = i+1;
                        end
                        ind = i;
                        
                        x = strfind(C{1,1}, '---');
                        i = ind;
                        while isempty(x{i})
                            i = i+1;
                        end
                        first = i+1;
                        
                        i = first;
                        while isempty(x{i})
                            i = i+1;
                        end
                        last = i-1;
                        
                        N = length(first:last);
                        for i = 1 : N
                            ind = first+i-1;
                            sigma(i,1) = str2num(cell2mat(C{1,1}(ind)));
                            sigma(i,2) = str2num(cell2mat(C{1,2}(ind)));
                            
                        end
                        
                        obj.ela{j}{k} = sigma;
                        next = last + 2 ;
                    end
                    
                end
                
            end
            
            %%
            
            fclose(fid);
            
        end
        
        % =====================================================================
        %> @brief loads excitation cross sections from LXcat-files
        %> Parameters: [exc,excThresh]
        % =====================================================================   
        function obj = excitation(obj)
            % loads excitation cross sections from LXcat-files 
            % into array exc and excThresh threshold energy into ionThresh
            
            for j = 1 : length(obj.dir)
                
                file = dir([obj.dir{j} '/*.txt']);
                fid = fopen([obj.dir{j} '/' file.name]);
                
                C = textscan(fid, '%s %s %s %s %s %s %s %s' );
                
                x  = strfind(C{1,1}, 'EXCITATION');
                xx = strfind(C{1,1}, '---');
                K = sum(cell2mat(x));
                
                if K == 0
                    
                    obj.exc{j} = [];
                    
                else
                    
                    
                    next = 1 ;
                    for k = 1 : K
                        
                        sigma = 0;
                        
                        i = next ;
                        while  isempty(x{i})
                            i = i+1;
                        end
                        ind = i;
                        ind_exc = ind + 2;
                        
                        i = ind;
                        while isempty(xx{i})
                            i = i+1;
                        end
                        first = i+1;
                        
                        i = first;
                        while isempty(xx{i})
                            i = i+1;
                        end
                        last = i-1;
                        
                        N = length(first:last);
                        for i = 1 : N
                            ind = first+i-1;
                            sigma(i,1) = str2num(cell2mat(C{1,1}(ind)));
                            sigma(i,2) = str2num(cell2mat(C{1,2}(ind)));
                            
                        end
                        
                        obj.exc{j}{k} = sigma;
                        obj.excThresh{j}{k} = str2num(cell2mat(C{1,1}(ind_exc)));
                        
                        next = last + 1 ;
                    end
                    
                end
                
            end
            
            %%
            
            fclose(fid);
            
        end
        
        % =====================================================================
        %> @brief loads momentum transfer cross sections from LXcat-files
        %> Parameters: [ela]
        % =====================================================================
        function obj = momentum(obj)
            % loads momentum transfer  cross sections from LXcat-files 
            % into array mom
            for j = 1 : length(obj.dir)
                
                file = dir([obj.dir{j} '/*.txt']);
                fid = fopen([obj.dir{j} '/' file.name]);
                
                C = textscan(fid, '%s %s %s %s %s %s %s %s' );
                
                x = strfind(C{1,1}, 'MOMENTUM');
                K = sum(cell2mat(x));
                
                if K == 0
                    
                    obj.mom{j} = [];;
                    
                else
                    
                    
                    next = 1 ;
                    for k = 1 : K
                        sigma = 0;
                        i = next ;
                        while  isempty(x{i})
                            i = i+1;
                        end
                        ind = i;
                        
                        x = strfind(C{1,1}, '---');
                        i = ind;
                        while isempty(x{i})
                            i = i+1;
                        end
                        first = i+1;
                        
                        i = first;
                        while isempty(x{i})
                            i = i+1;
                        end
                        last = i-1;
                        
                        N = length(first:last);
                        for i = 1 : N
                            ind = first+i-1;
                            sigma(i,1) = str2num(cell2mat(C{1,1}(ind)));
                            sigma(i,2) = str2num(cell2mat(C{1,2}(ind)));
                            
                        end
                        
                        obj.mom{j}{k} = sigma;
                        next = last + 2 ;
                    end
                    
                end
                
            end
            
            %%
            
            fclose(fid);
            
        end
        
        % =====================================================================
        %> @brief loads effective momentum transfer cross sections from LXcat-files
        %> Parameters: [ela]
        % =====================================================================
        function obj = effective(obj)
            % loads effective cross sections from LXcat-files 
            % into array eff
            
            for j = 1 : length(obj.dir)
                
                file = dir([obj.dir{j} '/*.txt']);
                fid = fopen([obj.dir{j} '/' file.name]);
                
                C = textscan(fid, '%s %s %s %s %s %s %s %s' );
                
                x = strfind(C{1,1}, 'EFFECTIVE');
                K = sum(cell2mat(x));
                
                if K == 0
                    
                   obj.eff{j} = [];
                    
                else
                    
                    
                    next = 1 ;
                    for k = 1 : K
                        sigma = 0;
                        i = next ;
                        while  isempty(x{i})
                            i = i+1;
                        end
                        ind = i;
                        
                        x = strfind(C{1,1}, '---');
                        i = ind;
                        while isempty(x{i})
                            i = i+1;
                        end
                        first = i+1;
                        
                        i = first;
                        while isempty(x{i})
                            i = i+1;
                        end
                        last = i-1;
                        
                        N = length(first:last);
                        for i = 1 : N
                            ind = first+i-1;
                            sigma(i,1) = str2num(cell2mat(C{1,1}(ind)));
                            sigma(i,2) = str2num(cell2mat(C{1,2}(ind)));
                            
                        end
                        
                        obj.eff{j}{k} = sigma;
                        next = last + 2 ;
                    end
                    
                end
                
            end
            
            %%
            
            fclose(fid);
            
        end
        
        % =====================================================================
        %> @brief converts effective mometum transfer cross section to  
        % elastic cross section
        %> Parameters: [ela]
        % ===================================================================== 
        function obj = effective2elastic(obj)
            % converts effective mometum transfer cross section eff to  
            % elastic cross section ela
            
            % gas species
            for j = 1 : length(obj.dir)
                
                 if isempty(obj.ela{j}) & ~isempty(obj.eff{j})
                     
                % collision types
                for k = 1 : length(obj.eff{j})
                   
                        
                        energy = obj.eff{j}{k}(:,1);
                        sigma = obj.eff{j}{k}(:,2);
                        
                        % formula from Vahedi et al (1995) p.190
                        beta = 1/2*( energy.*log(1+energy) )./...
                            ( energy - log(1+energy) );
                        % for energy = 0 --> beta = 1:
                         beta(energy<1e-6) = 1;
                        
                        %sigma = beta.*sigma;
                        
                        % substract excitation cross sections
                        for l = 1 : length(obj.exc{j})
                            
                            energy_eff = obj.exc{j}{l}(:,1);
                            sigma_eff = obj.exc{j}{l}(:,2);
                            energy_eff = [energy_eff ; 1e6];
                            sigma_eff = [sigma_eff ; sigma_eff(end)];
                            sigma = sigma - interp1(energy_eff,sigma_eff,energy,'linear');
                            
                        end
                        
                        % substract ionization cross sections
                        for l = 1 : length(obj.ion{j})
                            
                            energy_eff = obj.ion{j}{l}(:,1);
                            sigma_eff = obj.ion{j}{l}(:,2);
                            energy_eff = [energy_eff ; 1e6];
                            sigma_eff = [sigma_eff ; sigma_eff(end)];
                            sigma = sigma - interp1(energy_eff,sigma_eff,energy,'linear');
                            
                        end
                        
                        obj.ela{j}{k}(:,1) = energy;
                        obj.ela{j}{k}(:,2) = max(0,sigma);
                        
                    end
                    
                end
                
            end
            
        end
             
         % =====================================================================
        %> @brief Converts momentum transfer cross section to  
        % elastic cross section
        %> Parameters: [ela]
        % ===================================================================== 
        function obj = momentum2elastic(obj)
            % converts momentum transfer mom cross section to  
            % elastic cross section ela
            
            % gas species
            for j = 1 : length(obj.dir)
                
                if isempty(obj.ela{j}) & ~isempty(obj.mom{j})
                    
                % collision types
                for k = 1 : length(obj.mom{j})
                    
                                           
                        energy = obj.mom{j}{k}(:,1);
                        sigma = obj.mom{j}{k}(:,2);
                        
                        % formula from Vahedi et al (1995) p.190
                        beta = 1/2*( energy.*log(1+energy) )./...
                            ( energy -log(1+energy) );
                        % for energy = 0 --> beta = 1:
                        beta(energy<1e-6) = 1;
                        
                        sigma = beta.*sigma;
                        
                        obj.ela{j}{k}(:,1) = energy;
                        obj.ela{j}{k}(:,2) = sigma;
                        
                    end
                    
                end
                
            end
            
        end
           
        % =====================================================================
        %> @brief creates over all energy from LXcat data
        %> Parameters: [energy]
        % ===================================================================== 
        function obj = getEnergy(obj)
            % creates over all vector energy from LXcat data
            
            % gas species
            for j = 1 : length(obj.dir)
                
                % collision types
                sigma = obj.att;
                for k = 1 : length(sigma{j});
                    
                    energy = sigma{j}{k}(:,1);
                    obj.energy = [obj.energy ; energy] ;
                    
                end
                
                sigma = obj.ela;
                for k = 1 : length(sigma{j});
                    
                    energy = sigma{j}{k}(:,1);
                    obj.energy = [obj.energy ; energy] ;
                    
                end
                
                sigma = obj.exc;
                for k = 1 : length(sigma{j});
                    
                    energy = sigma{j}{k}(:,1);
                    obj.energy = [obj.energy ; energy] ;
                    
                end
                
                sigma = obj.ion;
                for k = 1 : length(sigma{j});
                    
                    energy = sigma{j}{k}(:,1);
                    obj.energy = [obj.energy ; energy] ;
                    
                end
                
                sigma = obj.mom;
                for k = 1 : length(sigma{j});
                    
                    energy = sigma{j}{k}(:,1);
                    obj.energy = [obj.energy ; energy] ;
                    
                end
                    
                
                % find unique values and sort them
                obj.energy = unique(obj.energy);
                obj.energy = sort(obj.energy);
                
                
            end
            
        end
        
        % =====================================================================
        %> @brief Prepare data for fit by adding data points 
        %> at energy = 0 and max(energy) 
        %> Parameters: [ion,att,exc]
        % ===================================================================== 
        function obj = prepareForFit1(obj)
            % prepare data for fit by adding data points at energy = 0 
            % and max(energy) for arrays ion, att and exc
            
            E_max = max(obj.energy);
             
            obj.ion  = obj.addDataPoints(obj.ion ,E_max);
            obj.att  = obj.addDataPoints(obj.att ,E_max);
            obj.exc  = obj.addDataPoints(obj.exc ,E_max);  

        end
        
        % =====================================================================
        %> @brief prepare data for fit by adding data points 
        %> at energy = 0 and max(energy) 
        %> Parameters: [ela,mom,eff]
        % ===================================================================== 
        function obj = prepareForFit2(obj)
            % prepare data for fit by adding data points at energy = 0 
            % and max(energy) for arrays ela, mom and eff
           
            E_max = max(obj.energy);
            
            if ~isempty(obj.ela)
            obj.ela  = obj.addDataPoints(obj.ela,E_max);
            end
            
            if ~isempty(obj.mom)
            obj.mom  = obj.addDataPoints(obj.mom,E_max );
            end
            
            if ~isempty(obj.eff)
            obj.eff  = obj.addDataPoints(obj.eff,E_max );
            end
            
        end
        
        % =====================================================================
        %> @brief adds data point to cross section sigma at energy = 0 and
        % zero-data point at energy, if sigma is empty 
        % ===================================================================== 
        function [sigma] = addDataPoints(obj,sigma,energy)
            % adds data point to cross section sigma at energy = 0 
                  
            % gas species
            for j = 1 : length(obj.dir)

                if isempty(sigma{j})
                    sigma{j} = {[energy'  0*energy']};
                end
                
                % collision types
                for k = 1 : length(sigma{j});
                    
                    xfit = sigma{j}{k}(:,1);
                    yfit = sigma{j}{k}(:,2);
                    
                    % add addiitonal point at beginning:
                    if xfit(1) > 0
                        xfit = [0 ; xfit];
                        yfit = [yfit(1) ; yfit];
                    end
                     
                    sigma{j}{k} = [xfit  yfit] ;
                    
                end
                
            end
            
        end
        
        % =====================================================================
        %> @brief  sets ionization and excitation threshold energies to zero if 
        % not available  
        %> Parameters: [ionThresh,excThresh]
        % ===================================================================== 
        function obj = fillThresholds(obj)
            % sets ionization and excitation threshold energies ionThresh 
            % and excThresh to zero if not available  
            
            % gas species
            for j = 1 : length(obj.dir)

                % ionization energies:
                if isempty(obj.ion{j})
                    obj.ionThresh{j} = {0};
                end
                
                % ionization energies:
                if isempty(obj.exc{j})
                    obj.excThresh{j} = {0};
                end
                                
            end
            
        end
        
        % =====================================================================
        %> @brief  makes total cross section and corresponding energy 
        % for each species
        %> Parameters: [energy,tot]
        % ===================================================================== 
        function obj = totalXsection(obj)
            % makes total cross section tot and corresponding energy 
            % for each species
           
             for j = 1 : length(obj.dir)
                 
             sigma_ion = sumXsection(obj,obj.ion{j},obj.energy);
             sigma_att = sumXsection(obj,obj.att{j},obj.energy);
             sigma_ela = sumXsection(obj,obj.ela{j},obj.energy);
             sigma_exc = sumXsection(obj,obj.exc{j},obj.energy);
             
             obj.tot{j} = sigma_ion + sigma_att + sigma_ela + sigma_exc;
             
             end
 
        end
        
        % =====================================================================
        %> @brief makes sum of one type of cross section over one species
        % ===================================================================== 
        function sigma_sum = sumXsection(obj,sigma,energy)
            % makes sum of one type of cross section over one species
                     
                if isempty(sigma)
                    sigma_sum = 0;
                else
                    sigma_sum = 0;
                    for k = 1 : length(sigma)
                       
                        x = sigma{k}(:,1);
                        y = sigma{k}(:,2);
                        
                        if energy(end) > x(end)
                            
                            x = [x ; energy(end)];
                            y = [y ; y(end)     ];
                            
                        end
                            
                        sigma_add = interp1(x,y,energy);
                        if ~isnan(sigma_add)
                            sigma_sum = sigma_sum + sigma_add;
                        end
                        
                    end
                end
            
        end
        
        % =====================================================================
        %> @brief plots fitted cross sections from LXcat
        % ===================================================================== 
        function plotXsections(obj,textsize,xscale,yscale)
            %Plots fitted cross sections from LXcat
            
            if obj.interactive == 1
                % gas species
                for j = 1 : length(obj.dir)

                    [fig, ax] = obj.GetFigure(xscale,yscale);
                    hold on; grid on
                    Legend = [];
                    for i = 1 : length(obj.ion{j})
                        plot(obj.ion{j}{i}(:,1),obj.ion{j}{i}(:,2),'r.-','Linewidth',2);
                        Legend{end+1} = 'ionization';
                    end
                    
                    for i = 1 : length(obj.att{j})
                        plot(obj.att{j}{i}(:,1),obj.att{j}{i}(:,2),'b.-','Linewidth',2);
                        Legend{end+1} = 'attachment';
                    end
                    
                    for i = 1 : length(obj.exc{j})
                        plot(obj.exc{j}{i}(:,1),obj.exc{j}{i}(:,2),'black.-','Linewidth',2);
                        obj.text2function( obj.exc{j}{i}(:,1),obj.exc{j}{i}(:,2) , [ num2str(obj.excThresh{j}{i}) ' eV'] , 12 )
                        Legend{end+1} =  [ 'excitation ' num2str(obj.excThresh{j}{i}) ' eV'];
                    end
                    
                    for i = 1 : length(obj.ela{j})
                        plot(obj.ela{j}{i}(:,1),obj.ela{j}{i}(:,2),'m.','Linewidth',2);
                        Legend{end+1} = 'elastic';
                    end
                    
                    
                    for i = 1 : length(obj.mom{j})
                        if sum(obj.mom{j}{i}(:,2)) > 0
                            plot(obj.mom{j}{i}(:,1),obj.mom{j}{i}(:,2),'g-','Linewidth',2);
                            Legend{end+1} = 'momentum transfer';
                        end
                    end
                    
                    
                    for i = 1 : length(obj.eff{j})
                        if sum(obj.eff{j}{i}(:,2)) > 0
                            plot(obj.eff{j}{i}(:,1),obj.eff{j}{i}(:,2),'c.-','Linewidth',2);
                            Legend{end+1} = 'effective momentum transfer';
                        end
                    end
                    
                    plot(obj.energy,obj.tot{j},'--','Linewidth',2);
                    Legend{end+1} = 'total';
                    
                    xlabel('energy [eV]','fontsize',textsize)
                    ylabel('\sigma  [eV]','fontsize',textsize)
                    set(gca,'fontsize',textsize)
                    %title(['Xsections:' num2str(obj.dir{j})])
                    legend(Legend)
                    
                    title(['gas ' num2str(j)])
                    
                    uicontrol('String','Continue',...
                        'Callback','uiresume(gcbf)');
                    uiwait(gcf);
                    disp(['Cross sections gas ' num2str(j) ' ok']);
                    
                end
                
            end
        end
        
        % =====================================================================
        %> @brief get figure properties
        % ===================================================================== 
        function [figure1, axes1] = GetFigure(obj,xscale,yscale)
            % get figure properties

            % set default scales: linear
            if nargin < 3
                xscale = 'linear';
                yscale = 'linear';
            end
            %widthX = 800;
            %widthY = 450;
            fontsize = 14;
            linewidth = 0.5;
            figure1 = figure('PaperSize',[29.68 20.98],...
                'WindowStyle','docked');
                %'Position', [1200-widthX 680-widthY widthX widthY]
            
            % 'PaperOrientation','landscape',...
            axes1 = axes('Parent',figure1, ...
                'XScale',xscale, 'XMinorTick','off', ...
                'YScale',yscale,'YMinorTick','on',...
                'LineWidth',linewidth, 'FontSize',fontsize);
            
            box(axes1,'on'); grid on
            xlabel('time [ns]','LineWidth',linewidth,'FontSize',fontsize);
            ylabel('amplitude [A]','LineWidth',linewidth,'FontSize',fontsize);
        end
    
        % =====================================================================
        %> @brief writes string to x- and y-data in Figure 
        % ===================================================================== 
        function text2function(obj,x,y,string,textsize)
            % writes text "string" on the maximum of function y(x)
            
            y_max = max(y);
            y_max = y_max(1);
            x_max = x(y==max(y));
            x_max = x_max(1);
            text(x_max,y_max, string ,'fontsize',textsize)
            
        end
        
    end
    
end


