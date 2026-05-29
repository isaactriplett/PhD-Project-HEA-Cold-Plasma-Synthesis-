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
classdef MonteCarlo
        
        properties
            
            % electron mass
            me = 9.10938291e-31;
            % electron charge
            q0 = 1.60217657e-19;
            % Boltzmann constant
            kB = 1.3806488 *10^(-23);
            % Avogadro constant
            Na = 6.02214129*10^23;
            % electric constant
            epsilon0 = 8.854188e-12;
            
            % cross secion data created by class importLXcat
            Xsec
            % cell array of sumformula of gas species
            gas
            % cell array of mass of gas species (in kg)
            mgas
            % fractions of individual species in the gas as a vector
            mix
            
            % number of initial electrons used in MC calculation
            N0
            % number of initial electrons used for space charge calculation
            n0
            % pressure in Pascal
            p
            % voltage in V
            U
            % distance in m
            d
            % temperature in Kelvin
            Temp
            
            % gas density in m^-3
            N
            
            % E/N for homogeneous field in Td
            EN = [];
            % minimal E/N for inhomogeneous field in Td
            EN_min = [];
            % maximal E/N for inhomogeneous field in Td
            EN_max = [];
            
            
            % length in x direction
            Lx
            % length in y direction
            Ly
            % length in z direction
            Lz
            % number of cells in x direction
            nx = 80
            % number of cells in y direction
            ny = 90
            % number of cells in z direction
            nz = 100
            % x_vector
            x
            % y_vector
            y
            % z_vector
            z
            % x_meshgrid
            X
            % y_meshgrid
            Y
            % z_meshgrid
            Z
            % [nx x ny x nz]-matrix with zeros inside and ones outside the
            %boundary
            boundary
            
            % vector of initial mean position of initial gaussian distributed
            % electrons in x,y and z direction
            pos_xyz  = [0 0 0]
            % vector of initial broadening of initial gaussian distributed
            % electrons in x,y and z direction
            sigma_xyz = [0 0 0]
            
            
            % tolerance in error of drift velcocity (default: 1%)
            w_err = 0.001;
            % tolerance in error of diffusion constant (default: 1%)
            DN_err = 0.001;
            % maximum allowed number of electorns
            Ne_max = 1e6
            % number of collisions until equilibrium (default: 20e6)
            col_equ = 10e6
            % number of collisions at which simulation ends (default: 20e6)
            col_max = 1e100;
            % conserve (1) electron number after ionizatzion/attachment or not (0)
            conserve = 1
            % (1) isotropic, (0) non-isotropic scattering
            % according to Vahedi et al.
            iso = 1
            % energy sharing in ionizing collision
            W
            % maximum electron energy
            E_max
            % maximal collision frequency:
            nu_max
            % collision counter
            counter
            % checks end of simulation: End =1 stops the simulation
            End = 0;
            % euqilibrium time
            T_sst
            % line number in output file:
            line = 1
            % computation time:
            elapsedTime
            % plot (1) or do not plot data (0)
            interactive
            
            % current time
            t = []
            % current time step dt
            dt
            % sum of all times for all electrons:
            t_total = 0
            % current position of electrons, cations and anions (order important)
            r = {[] [] []}
            % current velocity of electrons
            v
            % current acceleration of electrons
            a
            % current time-integrated velocity
            v_int
            % current time-integrated velocity-squared
            v2_int
            % collision indices for elastic collision
            ind_ela
            % collision indices for excitation collision
            ind_exc
            % collision indices for ionization collision
            ind_ion
            % collision indices for attachment collision
            ind_att
            % total number of all real collisions that happend
            collisions = 0
            % column numbers of elastic collision
            col_ela
            % column numbers of excitation collision
            col_exc
            % column numbers of ionization collision
            col_ion
            % column numbers of attachment collision
            col_att
            % Mass is a vector of length(v),
            % the entries are the masses of the gas-species undergoing elastic
            % collisions, all other entries are zero
            Mass
            % Loss is a vector of length(v),
            % the entries are the energy losses due to excitation collisions,
            % all other entries are zero
            Loss
            
            %For each ionization which occurs, 1/(# of electrons) is added
            %to a_directL
            %For each attachment which occurs, 1/(# of electrons) is
            %subtracted from a_directL
            %At the end of the simulation, a_directL will be divided by the
            %distance travelled by the electron swarm in order to calculate
            %the ionization coefficient directly

            a_directL = 0

            fittingnu = 0

            % temporal mean data of electron swarm
            mean
            % bulk transport data
            bulk
            % flux transport data
            flux
            % reaction rates
            rates
            % energy data
            E
            
            % charge density
            rho
            % electric potential
            phi
            % electric field function in x-direction
            E_x
            % electric field function in y-direction
            E_y
            % electric field function in z-direction
            E_z
            
            
        end
        
        methods
            
            
            % =====================================================================
            %> @brief checks if sum of gas fractions is equal to 1
            %> if not the case: last entry of mix will be corrected
            %> [mix]
            % =====================================================================
            function obj = checkFractionSum(obj)
                % checks if sum of gas fractions is equal to 1.
                % If not the case: last entry of mix will be corrected.
                
                if sum(obj.mix) == 1
                    
                else
                    
                    obj.mix(end) = 1 - sum(obj.mix(1:end-1));
                    fprintf('Sum of mixing ratios (in mix) NOT equal to one! \n ')
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculates the molar weight of a substance (in a.u)
            % =====================================================================
            function MM = MolMass(obj,substance)
                % function MolMass.m
                % The function MolMass calculates the molar weight of a substance.
                %
                % function call: MM = MolMass(substance)
                %
                % substance is a string of the chemical formula of s substance.
                % example:	MM = MolMass('Fe2(SO4)3');
                %
                % substance can also be a vector of substances opened by '[' and divided by space, comma or semicolon.
                % examples:
                % 			MM = MolMass('[Fe2(SO4)3 CuSO4 NaOH]');
                % 			MM = MolMass('[H2SO4;H2O;P;Cl2]');
                % 			MM = MolMass('[C3H5(OH)3,C3H7OH,C12H22O11,NaCl]');
                %
                % To distinguish charched substances the symbols '+' and '-' can be used.
                % exampels:
                % 			MM = MolMass('Fe2+')  --->  MM = 55.8470		%it means one mol of Fe2+
                % 			MM = MolMass('Fe3+')  --->  MM = 55.8470		%it means one mol of Fe3+
                % 		but	MM = MolMass('Fe2')   --->  MM = 111.6940		%it means two moles of Fe
                %
                %
                % ultimate Date:     18.12.2009
                % version:           1.4
                % copyright:         E. Giebler, TU Dresden, IfA (1999-2009)
                
                for char_index = 1:length(substance),														% first syntax check
                    if isempty(findstr(substance(char_index),'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[];, +-".')),
                        error_message = ['Wrong symbol ' substance(char_index) ' in the formula!']
                        error(error_message)
                    end%if
                end%for
                
                substance_number = 1;
                single_substance = '';
                char_index=0;
                
                % ---------------------replace redundant spaces--------------------------
                if   ~isempty(findstr('  ',substance))...
                        | ~isempty(findstr(' ]',substance))...
                        | ~isempty(findstr('[ ',substance))
                    while ~isempty(findstr('  ',substance))
                        substance = strrep(substance,'  ',' ');
                    end % while  ~isempty(findstr('  ',substance))
                    substance = strrep(substance,' ]',']');
                    substance = strrep(substance,'[ ','[');
                end %if findstr(...
                
                if substance(1) == '[',																		% vector of substances
                    char_index = char_index + 1;
                    while char_index < length(substance)
                        char_index = char_index + 1;
                        if ~isempty(findstr(substance(char_index),'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()+-".'))
                            single_substance(end+1) = substance(char_index);
                        elseif ~isempty(findstr(substance(char_index),' ,;]'))
                            MM(substance_number) = MolMass(single_substance);
                            single_substance = '';
                            substance_number = substance_number+1;
                            while char_index < length(substance) &  ~isempty(findstr(substance(char_index+1),' ,;]'))
                                char_index = char_index + 1;
                            end%while
                            if substance(char_index) == ']'
                                char_index = length(substance);
                            end%if
                        end%if
                    end%while
                    %  MM(substance_number) = MolMass(single_substance);
                else														                                % single substance only
                    
                    % molar masses of the elements:
                    H = 1.00794; He = 4.002602; Li = 6.941; Be = 9.012182; B = 10.811; C = 12.011; N = 14.00674;
                    O = 15.9994; F = 18.9984032; Ne = 20.1797; Na = 22.989768; Mg = 24.305; Al = 26.981539; Si = 28.0855;
                    P = 30.973762; S = 32.066; Cl = 35.4527; Ar = 39.948; K = 39.0983; Ca = 40.078; Sc = 44.95591;
                    Ti = 47.88; V = 50.9415; Cr = 51.9961; Mn = 54.93805; Fe = 55.847; Co = 58.9332; Ni = 58.69;
                    Cu = 63.546; Zn = 65.39; Ga = 69.723; Ge = 72.61; As = 74.92159; Se = 78.96; Br = 79.904; Kr = 83.8;
                    Rb = 85.4678; Sr = 87.62; Y = 88.90585; Zr = 91.224; Nb = 92.90638; Mo = 95.94; Tc = 98.9063;
                    Ru = 101.07; Rh = 102.9055; Pd = 106.42; Ag = 107.8682; Cd = 112.411; In = 114.82; Sn = 118.71;
                    Sb = 121.75; Te = 127.6; I = 126.90447; Xe = 131.29; Cs = 132.90543; Ba = 137.327; La = 138.9055;
                    Ce = 140.115; Pr = 140.90765; Nd = 144.24; Pm = 146.9151; Sm = 150.36; Eu = 151.965; Gd = 157.25;
                    Tb = 158.92534; Dy = 162.5; Ho = 164.93032; Er = 167.26; Tm = 168.93421; Yb = 173.04; Lu = 174.967;
                    Hf = 178.49; Ta = 180.9479; W = 183.85; Re = 186.207; Os = 190.2; Ir = 192.22; Pt = 195.08;
                    Au = 196.96654; Hg = 200.59; Tl = 204.3833; Pb = 207.2; Bi = 208.98037; Po = 208.9824; At = 209.9871;
                    Rn = 222.0176; Ac = 223.0197; Th = 226.0254; Pa = 227.0278; U = 232.0381; Np = 231.0359; Pu = 238.0289;
                    Am = 237.0482; Cm = 244.0642; Bk = 243.0614; Cf = 247.0703; Es = 247.0703; Fm = 251.0796; Md = 252.0829;
                    No = 257.0951; Lr = 258.0986; Rf = 259.1009; Db = 260.1053; Sg = 261.1087; Bh = 262.1138; Hs = 263.1182;
                    Mt = 262.1229;
                    Nn	= 1;	%Nn = Not named - for not named substances
                    
                    if substance(1) ~= '"',         % substance chemical formula (not a name)
                        
                        substance_index = 0;
                        bracket_index = 0;
                        last_bracket = [];
                        substance_char(1).second = '';
                        substance_char(1).first = '';
                        while char_index < length(substance),
                            char_index = char_index + 1;
                            if (substance(char_index)>= 'A') & (substance(char_index)<= 'Z'),
                                substance_index = substance_index+1;
                                factor(substance_index) = 1;
                                substance_mass(substance_index) = 0;
                                substance_char(substance_index).first = substance(char_index);
                                number = 0;
                            elseif (substance(char_index)>= 'a') & (substance(char_index)<= 'z'),		%small letter
                                substance_char(substance_index).second = substance(char_index);
                                number = 0;
                            elseif (substance(char_index)>= '0') & (substance(char_index)<= '9'),		%number
                                number = 1;
                                chartype(char_index,:) = 'nu';
                                if chartype(char_index-1,:) == 'nu',
                                    factor(substance_index) = factor(substance_index)*10+str2num(substance(char_index));
                                else%if
                                    factor(substance_index) = str2num(substance(char_index));
                                end%if
                            elseif (substance(char_index)== '+')|(substance(char_index)== '-'),
                                factor(substance_index) = 1;
                            elseif substance(char_index)== '(',
                                substance_index = substance_index+1;
                                bracket_level = 1;
                                bracket_char_index = 0;
                                while bracket_level > 0,
                                    char_index = char_index+1;
                                    bracket_char_index = bracket_char_index+1;
                                    if 		substance(char_index) == '(', bracket_level = bracket_level + 1;
                                    elseif	substance(char_index) == ')', bracket_level = bracket_level - 1;
                                    end%if
                                    if bracket_level > 0,
                                        bracket_substance(bracket_char_index) = substance(char_index);
                                    end%if
                                end%while
                                substance_char(substance_index).first = '(';
                                substance_mass(substance_index) = MolMass(bracket_substance);
                                number = 0;
                            end%if
                        end%while char_index
                        
                        for ei = 1:substance_index,
                            if substance_char(ei).first ~= '(',
                                if isempty(substance_char(ei).second),
                                    substance_mass(ei) = eval(substance_char(ei).first);
                                else
                                    substance_mass(ei) = eval([substance_char(ei).first;substance_char(ei).second]);
                                end% if
                            end%if
                        end%for
                        MM = substance_mass*factor';
                        
                    elseif any(substance=='('),         % substance is not a chemical formula but a name
                        for i = 2:length(substance),
                            if substance(i) == '(',
                                j = i+1; MM_str = '';
                                while substance(j) ~= ')' & j <= length(substance),
                                    MM_str = [MM_str substance(j)];
                                    j = j+1;
                                end % while
                                MM = str2num(MM_str);
                            end % if
                        end % for
                    else
                        MM = 1;
                    end % if substance(1) ~= '"',         % substance is not a chemical formula but a name
                    
                end % if substance(1) == '[',		  % vector of substances
                
            end
            
            % =====================================================================
            %> @brief converts mass (mgas) for the gas species from a.u. into kg
            %> Parameters: [mgas]
            % =====================================================================
            function obj = mass_in_kg(obj)
                % converts mass (mgas) for the gas species from a.u. into kg.
                
                for j = 1 : length(obj.gas)
                    
                    obj.mgas{j} = obj.MolMass(obj.gas{j})/obj.Na/1000;
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculates the gas number density (in m^-3) by the ideal
            %> gas law from pressure p and temperature Temp
            %> Parameters: [N]
            % =====================================================================
            function obj = gasNumberDensity(obj)
                % calculates the gas number density N (in m^-3) by the ideal gas law
                % from pressure p and temperature Temp.
                
                obj.N = obj.p/(obj.kB*obj.Temp);
                
            end
            
            % =====================================================================
            %> @brief calculates absolute value of velocity and energy in eV
            % =====================================================================
            function [abs_v E_in_eV] = velocity2energy_in_ev(obj,v)
                % calculates absolute value of velocity abs_v and energy
                % E_in_eV in eV.
                
                abs_v = sqrt(sum(v.^2,2));
                E_in_eV = 1/2*obj.me*abs_v.^2/obj.q0;
                
            end
            
            % =====================================================================
            %> @brief calculates maximal collision rate for gas mixture (in s^-1)
            %> Parameters: [nu_max]
            % =====================================================================
            function obj = maximalCollFreq(obj)
                % calculates maximal collision rate nu_max for gas mixture (in s^-1).
                
                % total cross secion of mixture:
                sigma = 0;
                for i = 1 : length(obj.gas)
                    sigma = sigma + obj.mix(i)*obj.Xsec.tot{i};
                end
                
                % add addtitonal point if necessary at E_max
                if obj.E_max > obj.Xsec.energy(end)
                    energy = [obj.Xsec.energy ; obj.E_max];
                    sigma =  [sigma ; sigma(end)];
                    
                else
                    energy = obj.Xsec.energy;
                end
                
                % velocity corresponding to energy:
                ve = sqrt(2*energy*obj.q0/obj.me);
                
                % maximal collision rate:
                obj.nu_max = max(obj.N.*sigma.*ve);
                
            end
            
            % =====================================================================
            %> @brief sets initial position and velocity of electrons
            %> Parameters: [r,v,mean]
            % =====================================================================
            function obj = initialParticles(obj)
                % sets time t to zero
                % sets initial electron number to N0
                % sets initial position of electrons r as gaussian distributed given by a mena position pos_xyz and sigma_xyz.
                % sets initial velocity of electrons v to zero
                
                obj.t = 0;
                obj.mean.energy = 0;
                obj.mean.position = obj.pos_xyz';
                obj.mean.sigma = obj.sigma_xyz';
                obj.mean.velocity = [0 0 0]';
                
                obj.mean.particles{1}(1) = obj.N0;
                obj.mean.particles{2}(1) = 0;
                obj.mean.particles{3}(1) = 0;
                
                obj.r{1}(:,1) = obj.pos_xyz(1) + obj.sigma_xyz(1)*randn(obj.N0,1) ;
                obj.r{1}(:,2) = obj.pos_xyz(2) + obj.sigma_xyz(2)*randn(obj.N0,1) ;
                obj.r{1}(:,3) = obj.pos_xyz(3) + obj.sigma_xyz(3)*randn(obj.N0,1) ;
                
                obj.v = zeros(size(obj.r{1}));
                % add noise to v (necessary to avoid singularities in
                % function elasticCollision):
                obj.v = obj.v + max(max(abs(obj.v)))*1e-6*rand(size(obj.v));
                
            end
            
            % =====================================================================
            %> @brief makes a 3d-meshgrid and defines boundaries
            %> Parameters: [x,y,z,X,Y,Z,boundary]
            % =====================================================================
            function obj = geometry(obj)
                % makes a 3d-meshgrid and defines boundaries
                
                obj.x = linspace(0, obj.Lx, obj.nx);
                obj.y = linspace(0, obj.Ly, obj.ny);
                obj.z = linspace(0, obj.Lz, obj.nz);
                
                [obj.X,obj.Y,obj.Z] = meshgrid(obj.x,obj.y,obj.z);
                %             obj.X = reshape(obj.X,obj.nx,obj.ny,obj.nz);
                %             obj.Y = reshape(obj.Y,obj.nx,obj.ny,obj.nz);
                %             obj.Z = reshape(obj.Z,obj.nx,obj.ny,obj.nz);
                
                
                %             obj.boundary = zeros(size(obj.X));
                %
                %             obj.boundary(1,:,:) = 1;
                %             obj.boundary(:,1,:) = 1;
                %             obj.boundary(:,:,1) = 1;
                %
                %             obj.boundary(end,:,:) = 1;
                %             obj.boundary(:,end,:) = 1;
                %             obj.boundary(:,:,end) = 1;
                
                
            end
            
            % =====================================================================
            %> @brief removes electrons outside the boundary
            %> Parameters: [Ne,r,v]
            % =====================================================================
            function obj = surfaceInteraction(obj)
                % removes electrons outside the boundary
                
                ind = find(...
                    obj.r{1}(:,1) < min(obj.x) | ...
                    obj.r{1}(:,1) > max(obj.x) | ...
                    obj.r{1}(:,2) < min(obj.y) | ...
                    obj.r{1}(:,2) > max(obj.y) | ...
                    obj.r{1}(:,3) < min(obj.z) | ...
                    obj.r{1}(:,3) > max(obj.z) );
                
                
                if ~isempty(ind)
                    
                    obj.v(ind,:) = [];
                    obj.r{1}(ind,:) = [];
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculates electron number density (in 1/m^3)
            %> Parameters: [rho]
            % =====================================================================
            function obj = particle2density(obj)
                % calculates electron number density rho (in 1/m^3)
                
                dx = obj.x(2) - obj.x(1);
                dy = obj.y(2) - obj.y(1);
                dz = obj.z(2) - obj.z(1);
                
                for i = 1 : length(obj.r)
                    
                    x = interp1(obj.x, 0.5:numel(obj.x)-0.5, obj.r{1}(:,1), 'nearest');
                    y = interp1(obj.y, 0.5:numel(obj.y)-0.5, obj.r{1}(:,2), 'nearest');
                    z = interp1(obj.z, 0.5:numel(obj.z)-0.5, obj.r{1}(:,3), 'nearest');
                    
                    % [x y]: Y then X because the plotting commands take matrices that look
                    % like Z(y-coord, x-coord)
                    if isempty(obj.r{i})
                        obj.rho{i}=0;
                    else
                        obj.rho{i} = accumarray([y x z] + 0.5, 1, [obj.ny obj.nx obj.nz]);
                        obj.rho{i} = obj.rho{i}/dx/dy/dz;
                    end
                end
                
            end
            
            % =====================================================================
            %> @brief solves 3D-Poisson equation iterativley for intitial phi0
            %> Parameters: [phi]
            % =====================================================================
            function obj = solvePoisson_3D(obj,niter,phi0)
                % solves 3D-Poisson equation for phi iterativley for intitial phi0
                % uses niter iterations
                
                if isempty(obj.EN)
                    
                    nx = obj.nx;
                    ny = obj.ny;
                    nz = obj.nz;
                    
                    X = obj.X;
                    Y = obj.Y;
                    Z = obj.Z;
                    
                    Lx = obj.Lx;
                    Ly = obj.Ly;
                    Lz = obj.Lz;
                    
                    dx = obj.x(2) - obj.x(1);
                    dy = obj.y(2) - obj.y(1);
                    dz = obj.z(2) - obj.z(1);
                    
                    b = zeros(ny,nx,nz); % x and y are reversed due to meshgrid difiniton
                    pn= zeros(ny,nx,nz);
                    
                    %starting potential
                    p = phi0;% obj.U/obj.d*obj.Z;
                    
                    i=2:ny-1;
                    j=2:nx-1;
                    k=2:nz-1;
                    
                    %Source term
                    rho = obj.rho{1} - obj.rho{2} + obj.rho{3};
                    b = obj.n0/obj.N0*obj.q0*rho/obj.epsilon0;
                    
                    %Explicit iterative scheme with C.D in space (5-point difference)
                    for it=1:niter
                        pn=p;
                        
                        p(i,j,k) = (...
                            dy^2*dz^2*( pn(i+1,j,k) + pn(i-1,j,k) ) + ...
                            dx^2*dz^2*( pn(i,j+1,k) + pn(i,j-1,k) ) + ...
                            dx^2*dy^2*( pn(i,j,k+1) + pn(i,j,k-1) ) + ...
                            dx^2*dy^2*dz^2 *b(i,j,k) )./...
                            ( 2*(dx^2*dy^2 + dx^2*dz^2 + dy^2*dz^2) );
                        
                        %Boundary conditions
                        %p(1,:,:)= 0;
                        %p(end,:,:)= 0;
                        %p(:,1,:)= 0;
                        %p(:,end,:)= 0;
                        p(:,:,1)= 0;
                        p(:,:,end)= obj.U;
                        
                        %                                     surf(p(:,:,obj.nz/2))
                        %                                     drawnow
                        
                    end
                    
                    % electric field:
                    [E_x,E_y,E_z] = gradient(p);
                    obj.E_x = E_x/dx;
                    obj.E_y = E_y/dy;
                    obj.E_z = E_z/dz;
                    
                    obj.phi = p;
                    
                    E_abs = sqrt(obj.E_x.^2 + obj.E_y.^2 + obj.E_z.^2);
                    obj.EN_min = min(min(min((E_abs))))/obj.N*1e21;
                    obj.EN_max = max(max(max((E_abs))))/obj.N*1e21;
                    
                else
                    
                    obj.E_x = 0;
                    obj.E_y = 0;
                    obj.E_z = obj.EN*obj.N*1e-21;
                    
                end
                
            end
            
            % =====================================================================
            %> @brief performs non-collissional flight for electrons in electric field
            %> Parameters: [t,dt,r,dr_dt,v,v_int,v2_int,E_in_eV,E_x,E_y,E_z,counter]
            % =====================================================================
            function obj = freeFlight(obj)
                % performs non-collissional flight for electrons in electric field
                
                % choose time step
                P_null = obj.randomNumbers(1);
                obj.dt = -(obj.nu_max)^-1*log(P_null);
                
                % make time vector
                if isempty(obj.t)
                    obj.t(end+1) = obj.dt;
                else
                    obj.t(end+1) = obj.t(end) + obj.dt;
                end
                
                if isempty(obj.EN)
                    
                    % E-fields:
                    Ex = interp3(obj.X,obj.Y,obj.Z,obj.E_x,obj.r{1}(:,1),obj.r{1}(:,2),obj.r{1}(:,3),'linear');
                    Ey = interp3(obj.X,obj.Y,obj.Z,obj.E_y,obj.r{1}(:,1),obj.r{1}(:,2),obj.r{1}(:,3),'linear');
                    Ez = interp3(obj.X,obj.Y,obj.Z,obj.E_z,obj.r{1}(:,1),obj.r{1}(:,2),obj.r{1}(:,3),'linear');
                    
                else
                    
                    Ex = 0*obj.v(:,1);
                    Ey = 0*obj.v(:,1);
                    Ez = obj.E_z*ones(size(Ex));
                    
                end
                
                E = [Ex Ey Ez];
                
                %acceleration due to electric field:
                a = obj.q0/obj.me*E;
                
                if ~isempty(obj.T_sst)
                    
                    ind = find(obj.t >= obj.T_sst);
                    
                    % integrated velocity
                    obj.v_int = obj.v*obj.dt + a/2*obj.dt^2;
                    % integrated velocity-squared
                    obj.v2_int = obj.v.^2*obj.dt +  a.*obj.v*obj.dt^2 + 1/3*a.^2*obj.dt^3;
                    
                end
                
                %new space and velocity:
                obj.r{1} = obj.r{1} +  obj.v*obj.dt + a/2*obj.dt^2;
                obj.v = obj.v + a*obj.dt;
                
                obj.counter = obj.counter + 1;
                
                
            end
            
            % =====================================================================
            %> @brief gets electron collective data: mean position, mean
            %> broadening in x,y and-direction, mean kinetic energy, electron number
            %> and total electron current
            %> Parameters: [mean]
            % =====================================================================
            function obj = collectMeanData(obj)
                % calculates electron collective data:
                % mean position, mean broadening in x,y and-direction, mean kinetic energy, electron number and total electron current
                
                [abs_v,E_in_eV ] = obj.velocity2energy_in_ev(obj.v);
                
                obj.mean.energy(end+1) = mean(E_in_eV);
                obj.mean.position(1:3,end+1) = mean(obj.r{1},1);
                obj.mean.velocity(1:3,end+1) = mean(obj.v,1);
                obj.mean.sigma(1:3,end+1) = std(obj.r{1},1,1);
                obj.mean.particles{1}(end+1) = obj.mean.particles{1}(end);
                obj.mean.particles{2}(end+1) = obj.mean.particles{2}(end);
                obj.mean.particles{3}(end+1) = obj.mean.particles{3}(end);
                
            end
            
            % =====================================================================
            %> @brief calculates mean energy and EEDF data after steady state was reached
            %> Parameters: [E]
            % =====================================================================
            function obj = energyData(obj)
                % calculates mean energy and EEDF data after steady state was reached
                
                if ~isempty(obj.T_sst)
                    
                    ind = find(obj.t >= obj.T_sst);
                    
                    if length(ind) > 10
                        
                        % sum of all times for all electrons:
                        obj.t_total = obj.t_total + obj.dt*size(obj.v,1);
                        
                        E_int = 1/2*obj.me/obj.q0*sum(obj.v2_int,2);
                        E_in_eV = E_int/obj.dt;
                        
                        %mean energy
                        obj.E.E_sum = obj.E.E_sum + sum(E_int);
                        
                        obj.E.E_mean = obj.E.E_sum/obj.t_total;
                        
                        [~,E_in_eV] = obj.velocity2energy_in_ev(obj.v);
                        obj.E.EEPF_sum = obj.E.EEPF_sum + histc(E_in_eV(:),obj.E.energy(:));
                        
                        % normalize EEPF
                        dE = obj.E.energy(2) - obj.E.energy(1);
                        obj.E.EEPF = obj.E.EEPF_sum/sum(obj.E.EEPF_sum)/dE;
                        
                        % get EEDF
                        energy  = obj.E.energy;
                        if energy(1)==0
                            energy(1) = energy(2);
                        end
                        
                        obj.E.EEDF = obj.E.EEPF(:)./sqrt(energy(:));
                        
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculates flux data after steady state was reached
            %> Parameters: [flux]
            % =====================================================================
            function  obj = fluxData(obj)
                % calculates flux data after steady state was reached
                
                if ~isempty(obj.T_sst)
                    
                    ind = find(obj.t >= obj.T_sst);
                    
                    if length(ind) > 10
                        
                        % drift velocity
                        obj.flux.v_int_sum = obj.flux.v_int_sum + sum(obj.v_int,1);
                        obj.flux.w = obj.flux.v_int_sum/obj.t_total;
                        
                        % diffusion constant:
                        D_flux = mean(obj.r{1}.*obj.v) - mean(obj.r{1}).*mean(obj.v);
                        obj.flux.N = obj.flux.N + 1;
                        obj.flux.D_sum = obj.flux.D_sum + D_flux;
                        obj.flux.DN = obj.N*obj.flux.D_sum/obj.flux.N;
                        
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculates bulk data after steady state was reached
            %> Parameters: [bulk]
            % =====================================================================
            function  obj = bulkData(obj)
                % calculates bulk data after steady state was reached
                
                if ~isempty(obj.T_sst)
                    
                    ind = find(obj.t > obj.T_sst);
                    
                    if length(ind) > 10
                        
                        x = obj.t(ind)' - obj.t(ind(1));
                        X = [ones(size(x)),x/max(x)];
                        
                        % drift in x-direction
                        y = obj.mean.position(1,ind)' - obj.mean.position(1,ind(1));
                        [b,bint] = regress(y/max(y),X);
                        obj.bulk.w(1) = b(2)*max(y)/max(x);
                        obj.bulk.w_err(1) = 0.25*diff(bint(2,:))*max(y)/max(x);
                        
                        % drift in y-direction
                        y = obj.mean.position(2,ind)' - obj.mean.position(2,ind(1));
                        [b,bint] = regress(y/max(y),X);
                        obj.bulk.w(2) = b(2)*max(y)/max(x);
                        obj.bulk.w_err(2) = 0.25*diff(bint(2,:))*max(y)/max(x);
                        
                        % drift in z-direction
                        y = obj.mean.position(3,ind)' - obj.mean.position(3,ind(1));
                        [b,bint] = regress(y/max(y),X);
                        obj.bulk.w(3) = b(2)*max(y)/max(x);
                        obj.bulk.w_err(3) = obj.N*0.25*diff(bint(2,:))*max(y)/max(x);
                        
                        % diffusion in x-direction
                        y = 0.5*obj.mean.sigma(1,ind).^2' - 0.5*obj.mean.sigma(1,ind(1))^2;
                        [b,bint] = regress(y/max(y),X);
                        obj.bulk.DN(1) = obj.N*b(2)*max(y)/max(x);
                        obj.bulk.DN_err(1) = obj.N*0.25*diff(bint(2,:))*max(y)/max(x);
                        
                        % diffusion in y-direction
                        y = 0.5*obj.mean.sigma(2,ind).^2' - 0.5*obj.mean.sigma(2,ind(1))^2;
                        [b,bint] = regress(y/max(y),X);
                        obj.bulk.DN(2) = obj.N*b(2)*max(y)/max(x);
                        obj.bulk.DN_err(2) = obj.N*0.25*diff(bint(2,:))*max(y)/max(x);
                        
                        % diffusion in z-direction
                        y = 0.5*obj.mean.sigma(3,ind).^2' - 0.5*obj.mean.sigma(3,ind(1))^2;
                        [b,bint] = regress(y/max(y),X);
                        obj.bulk.DN(3) = obj.N*b(2)*max(y)/max(x);
                        obj.bulk.DN_err(3) = 0.25*diff(bint(2,:))*max(y)/max(x);
                        
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculates the reaction rates by regression of electron number
            %> vs time
            %> Parameters: [rates]
            % =====================================================================
            function obj = rateDataCount(obj)
                % calculates the reaction rates by regression of electron number vs time
                
                if ~isempty(obj.T_sst)
                    
                    ind = find(obj.t >= obj.T_sst);

                    
                    if length(ind) > 10
                        
                        % electron number not conserved
                        if obj.conserve == 0
                            
                            if length(ind) > 1
                                
                                % find linear time interval:
                                x = obj.t(ind)'- obj.t(ind(1));
                                
                                % effective ionization:
                                y = log(obj.mean.particles{1}(ind)') - log(obj.mean.particles{1}(ind(1)));
                                obj.rates.count.eff = mean(y(2:end)./x(2:end))/obj.N;
                                obj.rates.count.eff_err = std(y(2:end)./x(2:end))/sqrt(length(ind))/obj.N;

                                
                                nu_eff = obj.rates.count.eff*obj.N;
                                x = ( exp(nu_eff*x) - 1 )/nu_eff *obj.mean.particles{1}(ind(1));
                                
                                % ionization:
                                y = obj.mean.particles{2}(ind)' - obj.mean.particles{2}(ind(1));
                                obj.rates.count.ion_tot = mean(y(2:end)./x(2:end))/obj.N;
                                obj.rates.count.ion_tot_err = std(y(2:end)./x(2:end))/sqrt(length(ind))/obj.N;
                                obj.rates.count.ion_tot(isnan(obj.rates.count.ion_tot)) = 0;
                                obj.rates.count.ion_tot(isnan(obj.rates.count.ion_tot_err)) = 0;
                                
                                % attachment:
                                y = obj.mean.particles{3}(ind)' - obj.mean.particles{3}(ind(1));
                                obj.rates.count.att_tot = mean(y(2:end)./x(2:end))/obj.N;
                                obj.rates.count.att_tot_err = std(y(2:end)./x(2:end))/sqrt(length(ind))/obj.N;
                                obj.rates.count.att_tot(isnan(obj.rates.count.att_tot)) = 0;
                                obj.rates.count.att_tot_err(isnan(obj.rates.count.att_tot_err)) = 0;
                                
                            end
                            
                        else % electron number conserved
                            
                            if length(ind) > 2
                                
                                x = obj.t(ind)'- obj.t(ind(1));
                                
                                % effective ionization:
                                y = ( obj.mean.particles{1}(ind)' - obj.mean.particles{1}(ind(1)) )/obj.mean.particles{1}(1) ;
                                obj.rates.count.eff = mean(y(2:end)./x(2:end))/obj.N;
                                obj.rates.count.eff_err = std(y(2:end)./x(2:end))/sqrt(length(ind))/obj.N;

                                
                                
                                % ionization:
                                y = ( obj.mean.particles{2}(ind)' - obj.mean.particles{2}(ind(1)) )/obj.mean.particles{1}(1) ;
                                obj.rates.count.ion_tot = mean(y(2:end)./x(2:end))/obj.N;
                                obj.rates.count.ion_tot_err = std(y(2:end)./x(2:end))/sqrt(length(ind))/obj.N;
                                obj.rates.count.ion_tot(isnan(obj.rates.count.ion_tot)) = 0;
                                obj.rates.count.ion_tot(isnan(obj.rates.count.ion_tot_err)) = 0;
                                
                                % attachment:
                                y = ( obj.mean.particles{3}(ind)' - obj.mean.particles{3}(ind(1)) )/obj.mean.particles{1}(1) ;
                                obj.rates.count.att_tot = mean(y(2:end)./x(2:end))/obj.N;
                                obj.rates.count.att_tot_err = std(y(2:end)./x(2:end))/sqrt(length(ind))/obj.N;
                                obj.rates.count.att_tot(isnan(obj.rates.count.att_tot)) = 0;
                                obj.rates.count.att_tot_err(isnan(obj.rates.count.att_tot_err)) = 0;
                                
                            end
                        end
                        
                    end
                    
                end
            end
            
            % =====================================================================
            %> @brief calcultes the reaction rates by the function convEEDFsigma
            %> Parameters: [rates]
            % =====================================================================
            function obj = rateDataConv(obj)
                % calcultes the reaction rates by the function rateDataConv
                
                if ~isempty(obj.T_sst)
                    
                    ind = find(obj.t >= obj.T_sst);
                    
                    if length(ind) > 10
                        
                        %% ionization rates:
                        Sigma = obj.Xsec.ion;
                        obj.rates.conv.ion_tot = 0; % sum of ionization rates
                        % species
                        for j = 1 : length(Sigma)
                            % collision types
                            for k = 1 : length(Sigma{j})
                                
                                obj.rates.conv.ion{j}{k} = obj.mix(j)*convEEDFsigma(obj,Sigma{j}{k});
                                obj.rates.conv.ion_tot = obj.rates.conv.ion_tot +  obj.rates.conv.ion{j}{k};
                                
                            end
                        end
                        %%
                        
                        %% attachment rates:
                        Sigma = obj.Xsec.att;
                        obj.rates.conv.att_tot = 0; % sum of ionization rates
                        % species
                        for j = 1 : length(Sigma)
                            % collision types
                            for k = 1 : length(Sigma{j})
                                
                                obj.rates.conv.att{j}{k} = obj.mix(j)*convEEDFsigma(obj,Sigma{j}{k});
                                obj.rates.conv.att_tot = obj.rates.conv.att_tot +  obj.rates.conv.att{j}{k};
                                
                            end
                        end
                        %%
                        
                        % effective rate:
                        obj.rates.conv.eff = obj.rates.conv.ion_tot - obj.rates.conv.att_tot;
                        
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief calculte the reaction rate by convolution between EEDF
            %> and cross section
            % =====================================================================
            function rate = convEEDFsigma(obj,sigma)
                % calculte the reaction rate by convolution between EEDF and cross section
                
                x = sigma(:,1);
                y = sigma(:,2);
                
                % add zero energy:
                if x(1) > 0
                    y = [y(1) ; y ];
                    x = [0    ; x ];
                end
                
                % add large(1e10 eV) energy:
                y = [y ; y(end)];
                x = [x ; 1e10 ];
                
                sigma_f = interp1(x,y,obj.E.energy,'linear');
                
                EEPF = obj.E.EEPF;
                x = obj.E.energy;
                dx = obj.E.energy(2) - obj.E.energy(1);
                
                A = ( EEPF(:).*x(:).^0.5)*dx;
                
                rate = sqrt(2*obj.q0/obj.me)*A(:)'*sigma_f(:);
                
            end
            
            
            % =====================================================================
            %> @brief decides which collision will happen for each electron
            %> Parameters: [ind_ela,ind_exc,ind_ion,ind_at,Mass,Loss]
            % =====================================================================
            function obj = collisionMatrix(obj)
                % decides which collision will happen for each electron
                
                [abs_v,E_in_eV] = obj.velocity2energy_in_ev(obj.v);
                C = []; mass=[]; loss = [];
                % elastic:
                [C N_ela col_ela mass loss] = obj.collisionfreq(abs_v,E_in_eV,'obj.Xsec.ela',C,mass,loss);
                % excitation:
                [C N_exc col_exc mass loss] = obj.collisionfreq(abs_v,E_in_eV,'obj.Xsec.exc',C,mass,loss);
                % ionization:
                [C N_ion col_ion mass loss] = obj.collisionfreq(abs_v,E_in_eV,'obj.Xsec.ion',C,mass,loss);
                % attachment:
                [C N_att col_att mass loss] = obj.collisionfreq(abs_v,E_in_eV,'obj.Xsec.att',C,mass,loss);
                
                
                %cumulative sum:
                C = cumsum(C,2);
                
                %check if elements > 1:
                if ~isempty(find(C > 1)) | ~isempty(find(isnan(C)))
                    fprintf('E_max might be too small! \n')
                    %obj.End = 1;
                end
                
                % last column of C is random vector R:
                N = size(obj.r{1},1);
                R = obj.randomNumbers([N 1]);
                R = repmat(R,1,size(C,2));
                
                % sort:
                ind = sum(R>C,2) + 1;
                
                obj.Mass = zeros(N,1);
                
                for i = 1 : length(col_ela)
                    index = find(ind == col_ela(i));
                    obj.Mass(index) = mass(col_ela(i));
                end
                
                obj.Loss = zeros(N,1);
                
                for i = 1 : length(col_exc)
                    index = find(ind == col_exc(i));
                    obj.Loss(index) = loss(col_exc(i));
                end
                
                for i = 1 : length(col_ion)
                    index = find(ind == col_ion(i));
                    obj.Loss(index) = loss(col_ion(i));
                end
                
                % indices that give electrons that perform certain collision:
                obj.ind_ela = find(ind >= col_ela(1) & ind <= col_ela(end));
                obj.ind_exc = find(ind >= col_exc(1) & ind <= col_exc(end));
                obj.ind_ion = find(ind >= col_ion(1) & ind <= col_ion(end));
                obj.ind_att = find(ind >= col_att(1) & ind <= col_att(end));
                
                % count all real collision that happend:
                col = length(obj.ind_ela)+length(obj.ind_exc)+...
                    length(obj.ind_ion)+length(obj.ind_att);
                %This calculation method for ioncol_normal (and thus
                %obj.a_directL) assumes that the number of electrons
                %created at a given timestep is small compared to the total
                %number of electrons
                %ioncol_normal = (length(obj.ind_ion)-length(obj.ind_att))/obj.mean.particles{1}(end);
                ioncol_normal = length(obj.ind_ion)/obj.mean.particles{1}(end);
                obj.collisions = obj.collisions + col;
                obj.a_directL(end+1) = obj.a_directL(end) + ioncol_normal;
                
            end
            
            % =====================================================================
            %> @brief calculates collision rates for columns of collision matrix
            % =====================================================================
            function [C N col mass loss] = collisionfreq(obj,abs_v,E_in_eV,sigma,C,mass,loss)
                % calculates collision rates for columns of collision matrix
                
                N = 0;
                col = [];
                % gas species
                Sigma = eval(sigma);
                for j = 1 : length(Sigma)
                    % collision types
                    for k = 1 : length(Sigma{j})
                        N = N +1;
                        x = Sigma{j}{k}(:,1);
                        y = Sigma{j}{k}(:,2);
                        
                        % add data point at the energy E_max
                        if obj.E_max > x(end)
                            x(end+1) = obj.E_max;
                            y(end+1) = y(end);
                        end
                        
                        sigma_f = interp1(x,y,E_in_eV,'linear');
                        C(:,end+1) = obj.mix(j)*obj.N*sigma_f.*abs_v/obj.nu_max;
                        col(end+1) = size(C,2);
                        mass(end+1) = obj.mgas{j};
                        
                        % excitation
                        if strcmp(sigma,'obj.Xsec.exc')
                            loss(end+1) = obj.Xsec.excThresh{j}{k};
                        end
                        
                        % ionization
                        if strcmp(sigma,'obj.Xsec.ion')
                            loss(end+1) = obj.Xsec.ionThresh{j}{k};
                        end
                        
                        % ionization
                        if strcmp(sigma,'obj.Xsec.att')
                            loss(end+1) = 0;
                        end
                        
                        % ionization
                        if strcmp(sigma,'obj.Xsec.ela')
                            loss(end+1) = 0;
                        end
                        
                    end
                end
                
                
                
            end
            
            % =====================================================================
            %> @brief performs elastic collision (isotropic or non-isotropic)
            %> Parameters: [v]
            % =====================================================================
            function obj = elasticCollision(obj)
                % performs elastic collision (isotropic or non-isotropic)
                
                ind = obj.ind_ela;
                
                if  ~isempty(ind)
                    
                    v = obj.v(ind,:);
                    
                    [~,E_in_eV] = obj.velocity2energy_in_ev(v);
                    
                    % initial eularian angles
                    [theta_1,phi_1,abs_v] = cart2sph(v(:,1),v(:,2),v(:,3));
                    
                    % unit vector of incident velocity
                    [x_1,y_1,z_1] = sph2cart(theta_1,phi_1,1);
                    e1 = [x_1,y_1,z_1];
                    
                    % randomly chosen scattering angle:
                    
                    phi = 2*pi*obj.randomNumbers(size(ind));
                    sin_phi = sin(phi);
                    cos_phi =  sqrt(1 - sin_phi.^2);
                    
                    if obj.iso == 1
                        cos_xsi = 1 - 2*obj.randomNumbers(size(ind));
                        xsi = acos(cos_xsi);
                    else
                        cos_xsi = ( 2 + E_in_eV - 2*(1+E_in_eV).^obj.randomNumbers(size(ind)) )./E_in_eV;
                        %R = obj.randomNumbers(size(ind));
                        %cos_xsi = 1 - 2*R - R.*(R-1).*E_in_eV;
                        xsi = acos(cos_xsi);
                    end
                    
                    sin_xsi = sign(randn(size(cos_xsi))).*sqrt(1-cos_xsi.^2);
                    
                    % unit vector in x-direction
                    e_x  = zeros(size(e1));
                    e_x(:,1) = 1;
                    
                    cos_theta = sum(e1.*e_x,2);
                    sin_theta = sqrt(1 - cos_theta.^2);
                    
                    % equation (11) in Vahedi et al (1995):
                    e2 = repmat(cos_xsi,1,3).*e1 + ...
                        repmat(sin_xsi.*sin_phi./sin_theta,1,3).*cross(e1,e_x) +...
                        repmat(sin_xsi.*cos_phi./sin_theta,1,3).*cross( e1 , cross(e_x,e1) );
                    % normalize:
                    e2 = e2./repmat(sqrt(sum(e2.^2,2)),1,3);
                    
                    % energy after collision, equation (12) in Vahedi et al (1995):
                    [abs_v,E_in_eV] = obj.velocity2energy_in_ev(obj.v(ind,:));
                    E_in_eV = max( 0,E_in_eV.*( 1-2*obj.me./obj.Mass(ind).*(1-cos(xsi)) ) );
                    abs_v = sqrt(2*E_in_eV*obj.q0/obj.me);
                    obj.v(ind,:) = repmat(abs_v,1,3).*e2;
                    
                    
                end
                
            end
            
            % =====================================================================
            %> @brief performs inelastic collision (isotropic or non-isotropic)
            %> Parameters: [v]
            % =====================================================================
            function obj = inelasticCollision(obj)
                % performs inelastic collision (isotropic or non-isotropic)
                
                ind = obj.ind_exc;
                
                if  ~isempty(ind)
                    
                    v = obj.v(ind,:);
                    
                    [~,E_in_eV] = obj.velocity2energy_in_ev(v);
                    
                    % initial eularian angles
                    [theta_1,phi_1,abs_v] = cart2sph(v(:,1),v(:,2),v(:,3));
                    
                    % unit vector of incident velocity
                    [x_1,y_1,z_1] = sph2cart(theta_1,phi_1,1);
                    e1 = [x_1,y_1,z_1];
                    
                    % randomly chosen scattering angle:
                    
                    phi = 2*pi*obj.randomNumbers(size(ind));
                    sin_phi = sin(phi);
                    cos_phi =  sqrt(1 - sin_phi.^2);
                    
                    if obj.iso == 1
                        cos_xsi = 1 - 2*obj.randomNumbers(size(ind));
                        xsi = acos(cos_xsi);
                    else
                        cos_xsi = ( 2 + E_in_eV - 2*(1+E_in_eV).^obj.randomNumbers(size(ind)) )./E_in_eV;
                        %R = obj.randomNumbers(size(ind));
                        %cos_xsi = 1 - 2*R - R.*(R-1).*E_in_eV;
                        xsi = acos(cos_xsi);
                    end
                    
                    sin_xsi = sign(randn(size(cos_xsi))).*sqrt(1-cos_xsi.^2);
                    
                    % unit vector in x-direction
                    e_x  = zeros(size(e1));
                    e_x(:,1) = 1;
                    
                    cos_theta = sum(e1.*e_x,2);
                    sin_theta = sqrt(1 - cos_theta.^2);
                    
                    % equation (11) in Vahedi et al (1995):
                    e2 = repmat(cos_xsi,1,3).*e1 + ...
                        repmat(sin_xsi.*sin_phi./sin_theta,1,3).*cross(e1,e_x) +...
                        repmat(sin_xsi.*cos_phi./sin_theta,1,3).*cross( e1 , cross(e_x,e1) );
                    % normalize:
                    e2 = e2./repmat(sqrt(sum(e2.^2,2)),1,3);
                    
                    % energy after collision
                    [abs_v,E_in_eV] = obj.velocity2energy_in_ev(obj.v(ind,:));
                    E_in_eV = max( 0, E_in_eV - obj.Loss(ind) );
                    abs_v = sqrt(2*E_in_eV*obj.q0/obj.me);
                    obj.v(ind,:) = repmat(abs_v,1,3).*e2;
                    
                    
                end
                
                
            end
            
            % =====================================================================
            %> @brief performs ionization collision for electrons
            %> Parameters: [v,r,mean]
            % =====================================================================
            function obj = ionCollision(obj)
                % performs ionization collision for electrons
                
                ind = obj.ind_ion;
                
                if  ~isempty(ind)
                    
                    v = obj.v(ind,:) + 0*max(max(abs(obj.v(ind,:))))*1e-6*rand(size(obj.v(ind,:)));
                    
                    [~,E_in_eV] = obj.velocity2energy_in_ev(v);
                    
                    % initial eularian angles
                    [theta_1,phi_1,abs_v] = cart2sph(v(:,1),v(:,2),v(:,3));
                    
                    % unit vector of incident velocity
                    [x_1,y_1,z_1] = sph2cart(theta_1,phi_1,1);
                    e1 = [x_1,y_1,z_1];
                    
                    % randomly chosen scattering angle:
                    
                    phi = 2*pi*obj.randomNumbers(size(ind));
                    sin_phi = sin(phi);
                    cos_phi =  sqrt(1 - sin_phi.^2);
                    
                    if obj.iso == 1
                        cos_xsi = 1 - 2*obj.randomNumbers(size(ind));
                        xsi = acos(cos_xsi);
                    else
                        cos_xsi = ( 2 + E_in_eV - 2*(1+E_in_eV).^obj.randomNumbers(size(ind)) )./E_in_eV;
                        %R = obj.randomNumbers(size(ind));
                        %cos_xsi = 1 - 2*R - R.*(R-1).*E_in_eV;
                        xsi = acos(cos_xsi);
                    end
                    
                    sin_xsi = sign(randn(size(cos_xsi))).*sqrt(1-cos_xsi.^2);
                    
                    % unit vector in x-direction
                    e_x  = zeros(size(e1));
                    e_x(:,1) = 1;
                    
                    cos_theta = sum(e1.*e_x,2);
                    sin_theta = sqrt(1 - cos_theta.^2);
                    
                    % equation (11) in Vahedi et al (1995):
                    e2 = repmat(cos_xsi,1,3).*e1 + ...
                        repmat(sin_xsi.*sin_phi./sin_theta,1,3).*cross(e1,e_x) +...
                        repmat(sin_xsi.*cos_phi./sin_theta,1,3).*cross( e1 , cross(e_x,e1) );
                    % normalize:
                    e2 = e2./repmat(sqrt(sum(e2.^2,2)),1,3);
                    
                    % energy after collision
                    [abs_v,E_in_eV] = obj.velocity2energy_in_ev(obj.v(ind,:));
                    %This simulation allows for impossible ionizations
                    %(where there is not enough kinetic energy for
                    %ionization)
                    E_in_eV = max( 0, E_in_eV - obj.Loss(ind) );
                    %E_in_eV = E_in_eV - obj.Loss(ind);
                    %Substituting with line below affected nothing
                    
                    % energy of incident electron:
                    abs_v = sqrt(2*obj.W*E_in_eV*obj.q0/obj.me);
                    obj.v(ind,:) = repmat(abs_v,1,3).*e2;
                    
                    % created electrons:
                    delta_Ne = length(ind); % number of created electrons
                    
                    abs_v = sqrt( 2*(1-obj.W)*E_in_eV*obj.q0/obj.me );
                    obj.v(end+1:end+delta_Ne,:)  = -repmat(abs_v,1,3).*e2;
                    obj.r{1}(end+1:end+delta_Ne,:) = obj.r{1}(ind,:);
                    
                    % electron creation:
                    obj.mean.particles{1}(end) = obj.mean.particles{1}(end) + delta_Ne;
                    
                    % cation creation
                    obj.mean.particles{2}(end) = obj.mean.particles{2}(end) + delta_Ne;
                    obj.r{2}(end+1:end+delta_Ne,:) = obj.r{1}(ind,:);
                    
                    % remove random electron:
                    if obj.conserve == 1
                        
                        % choose random electron from remaining ensemble:
                        i = randperm(length(obj.v),delta_Ne);
                        
                        % remove electrons
                        obj.v(i,:) = [];
                        obj.r{1}(i,:) = [];
                        
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief performs attachment collision for electrons
            %> Parameters: [v,r,mean]
            % =====================================================================
            function obj = attachCollision(obj)
                % performs attachment collision for electrons
                
                ind = obj.ind_att;
                
                if ~isempty(ind)
                    
                    % number of created anion
                    delta_Ne = length(ind);
                    
                    % electrons:
                    obj.mean.particles{1}(end) = obj.mean.particles{1}(end) - delta_Ne;
                    
                    % anions:
                    obj.mean.particles{3}(end) = obj.mean.particles{3}(end) + delta_Ne;
                    
                    % anion creation
                    obj.r{3}(end+1:end+delta_Ne,:) = obj.r{1}(ind,:);
                    
                    % remove electron
                    obj.v(ind,:) = [];
                    obj.r{1}(ind,:) = [];
                    
                    % create electron to conserve electron number
                    if obj.conserve == 1
                        
                        % choose random electron from remaining ensemble:
                        i = randperm(length(obj.v),delta_Ne);
                        
                        % add electrons
                        obj.v(end+1:end+delta_Ne,:) = obj.v(i,:);
                        obj.r{1}(end+1:end+delta_Ne,:) = obj.r{1}(i,:);
                        
                    end
                    
                end
                
                
            end
            
            % =====================================================================
            %> @brief checks if steady-state is reached
            %> Parameters: [T_sst,conuter,collisions,line]
            % =====================================================================
            function obj = checkSST(obj)
                % checks if steady-state is reached
                
                if isempty(obj.T_sst)
                    
                    if (obj.collisions/1e6) >= obj.line
                        
                        if obj.collisions >= obj.col_equ
                            
                            % check if the interval 80-90 % of energy data is larger
                            % than the interval90-100%:
                            n = length(obj.mean.energy);
                            n = round(n/10);
                            if mean(obj.mean.energy(end-2*n:end-n)) >= mean(obj.mean.energy(end-n:end))
                                
                                obj.T_sst = obj.t(end);
                                obj.counter = 0;
                                obj.collisions = 0;
                                obj.line = 1;
                                %obj.r{1} = ones(size(obj.r{1}));
                                
                            end
                            
                        end
                        
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief stops the simulation
            %> Parameters: [End]
            % =====================================================================
            function obj = endSimulation(obj)
                % stops the simulation
                
                % end simulation if too many electrons:
                if obj.conserve==0 & length(obj.v) > obj.Ne_max
                    obj.End = 1;
                    fprintf('simulation ended: maximum number of electrons reached \n')
                end
                
                % end simulation if no electrons:
                if obj.conserve==0 & length(obj.v) == 0
                    obj.End = 1;
                    fprintf('simulation ended: number of electrons is zero \n')
                end
                
                % end simulation if w_err < wz_err and D_err < Dz_err
                if ~isempty(obj.bulk)
                    if abs(obj.bulk.w_err(3)/obj.bulk.w(3)) < abs(obj.w_err) & abs(obj.bulk.DN_err(3)/obj.bulk.DN(3)) < abs(obj.DN_err)
                        obj.End = 1;
                        fprintf(['simulation ended: errors in w < %.2f %% and D < %.2f %% \n'],100*obj.w_err,100*obj.DN_err)
                    end
                end
                
                % end simulation if # of collisions > col_max:
                if obj.collisions > obj.col_max
                    obj.End = 1;
                    fprintf('simulation ended: maximum number of collisions reached \n')
                end
                
            end
            
            % =====================================================================
            %> @brief generates random numbers p
            % =====================================================================
            function p = randomNumbers(obj,size)
                % generates random numbers p
                
                %             q = haltonset(size(end),'Skip',1e3,'Leap',1e2);
                %             q = scramble(q,'RR2');
                %             ind = randi(size(1),size(1),size(end));
                %             p = q(ind);
                
                p = rand(size);
                
            end
            
            % =====================================================================
            %> @brief prints results in command window during simulation and
            %> saves results using getResults
            % =====================================================================
            function  obj = printOnScreen(obj)
                %  prints results in command window during simulation and saves results using getResults
                
                if (obj.collisions/1e6) >= obj.line
                    
                    obj.line = obj.line + 1;
                    
                    if ~isempty(obj.bulk)
                        
                        fprintf(['collisions: %i',...
                            ' electrons: %i',...
                            ' E: %.3e eV',...
                            ' w_bulk: %.3e m/s',...
                            ' w_flux: %.3e m/s',...
                            ' DN_bulk: %.2e (ms)^-1',...
                            ' DN_flux: %.2e (ms)^-1',...
                            ' Reff_count: %.2e m^3/s',...
                            ' Reff_calc: %.2e m^3/s',...
                            ' a_directL: %.2e',...
                            '\n'],...
                            obj.collisions,...
                            obj.mean.particles{1}(end),...
                            obj.E.E_mean,...
                            obj.bulk.w(3),...
                            obj.flux.w(3),...
                            obj.bulk.DN(3),...
                            obj.flux.DN(3),...
                            obj.rates.count.eff,...
                            obj.rates.conv.eff,...
                            obj.a_directL);
                        
                        % save data
                        obj.getResults();
                        
                    else
                        
                        fprintf([' collisions: %i',...
                            ' electrons: %i',...
                            ' mean energy: %.2e',...
                            '\n'],...
                            obj.collisions,...
                            obj.mean.particles{1}(end),...
                            obj.mean.energy(end) );
                        
                    end
                    
                    diary('temporal.txt');
                    
                    
                    
                end
                
                
            end
            
% =====================================================================
%> @brief plots temporal electron number, mean energy, mean position,
% mean width and EEDF
% =====================================================================
function  obj = plotMeanData(obj)
    % plots temporal electron number, mean energy, mean position, mean width and EEDF

    if (obj.collisions/1e6) >= obj.line && obj.interactive == 1

        close all

        %% ==================== FIGURE 1: Electron Number + Fit ====================
        if obj.conserve == 0
            scale = 'log';
        else
            scale = 'linear';
        end

        [fig1, axes1] = obj.Xsec.GetFigure('linear', scale); 
        hold on

        t_ns = 1e9 * obj.t;                    % time in ns
        Ne   = obj.mean.particles{1};          % number of electrons

        % Plot actual electron number
        plot(t_ns, Ne, 'b.', 'MarkerSize', 12, 'DisplayName', 'N_e (simulation)');

        % === Fit ln(N) = c*t + d  using ALL data points ===
        if length(Ne) > 5
            t_fit = obj.t';           % use entire time vector
            N_fit = Ne';

            % Linear fit: ln(N) = c*t + d
            p = polyfit(t_fit, log(N_fit), 1);   % p(1) = c (slope), p(2) = d (intercept)
            c = p(1);
            d = p(2);

            % Generate fitted curve
            N_fit_line = exp(c * t_fit + d);

            % Plot the fit on the same graph
            plot(t_ns, N_fit_line, 'r-', 'LineWidth', 2, ...
                'DisplayName', sprintf('Fit: ln(N) = %.4e * t + %.4f', c, d));

            %Record c
            obj.fittingnu = c;
        end

        ylabel('# of electrons')
        xlabel('time [ns]')
        title(['N_e = ' num2str(Ne(end))])
        legend('Location','best')
        grid on

        % ... (rest of your original plotting code for energy, position, etc. stays the same)
                    [fig6,axes6] = obj.Xsec.GetFigure('linear','linear'); hold on
                    plot(1e9*obj.t,obj.a_directL/obj.mean.position(3,:),'b.')
                    ylabel('ionization coefficient [m^(-1)]')
                    title('Directly calculated ionization coefficient')
                    
                    [fig2, axes2] = obj.Xsec.GetFigure() ; hold on
                    plot(1e9*obj.t,obj.mean.energy,'b.')
                    ylabel('energy [eV]')
                    title(['E/N=' num2str(obj.EN),' Td'...
                        ' collisions = ',num2str(round(obj.collisions/1e6)) ' 10^6'])
                    
                    if ~isempty(obj.T_sst)
                        y = [0.8*obj.mean.energy(end) 1.2*obj.mean.energy(end)];
                        line(1e9*[obj.T_sst obj.T_sst], y ...
                            , 'LineWidth',2,'LineStyle','-', 'Color','k');
                        text(1.1*1e9*obj.T_sst,y(2), 'T_{sst}'...
                            , 'HorizontalAlignment','center', 'VerticalAlignment','top', 'FontWeight','bold');
                    end
                    
                    [fig3, axes3] = obj.Xsec.GetFigure() ; hold on
                    plot(1e9*obj.t,obj.mean.position(1,:),'mo')
                    plot(1e9*obj.t,obj.mean.position(2,:),'r+')
                    plot(1e9*obj.t,obj.mean.position(3,:),'b.')
                    legend('x' , 'y' , 'z')
                    ylabel('position [m]')
                    
                    [fig4, axes4] = obj.Xsec.GetFigure() ; hold on
                    plot(1e9*obj.t,obj.mean.sigma(1,:).^2,'mo')
                    plot(1e9*obj.t,obj.mean.sigma(2,:).^2,'r+')
                    plot(1e9*obj.t,obj.mean.sigma(3,:).^2,'b.')
                    legend('x' , 'y' , 'z')
                    ylabel('\sigma^2 [m^2]')
                    
                    [fig5, axes5] = obj.Xsec.GetFigure() ; hold on
                    if obj.E.EEPF_sum == 0
                        [~,E_in_eV] = obj.velocity2energy_in_ev(obj.v);
                        EEPF = histc(E_in_eV,obj.E.energy);
                        plot(obj.E.energy,EEPF,'b-')
                        title(['<\epsilon> = ' num2str(obj.mean.energy(end))])
                    else
                        EEPF = obj.E.EEPF;
                        plot(obj.E.energy,EEPF,'b-')
                        title(['<\epsilon> = ' num2str(obj.E.E_mean)])
                    end
                    ind = find(EEPF > 0 ,1,'last');
                    x_max = obj.E.energy(ind);
                    xlim([0 max(0.1,x_max)]);
                    xlabel('\epsilon [eV]')
                    ylabel('EEPF')
                    
                    drawnow
                    
                    
                end
                
            end
            
            % =====================================================================
            %> @brief plots all charged particles at their position r
            % =====================================================================
            function plotParticles(obj)
                % plots all charged particles at their position r
                
                if mod(obj.counter,1) == 0;
                    
                    colors = {'b.' 'r.' 'black.'};
                    hold off
                    for i = 1  : length(obj.r)
                        if ~isempty(obj.r{i})
                            
                            plot3(1e3*obj.r{i}(:,1),1e3*obj.r{i}(:,2),1e3*obj.r{i}(:,3),colors{i})
                            view(0,0)
                            grid on
                            xlabel('x [mm]')
                            ylabel('y [mm]')
                            zlabel('z [mm]')
                            %                         xlim([0 1e3*obj.Lx])
                            %                         ylim([0 1e3*obj.Ly])
                            %                         zlim([0 1e3*obj.Lz])
                            drawnow
                            hold on
                            
                            
                        end
                    end
                    
                end
                
            end
            
            % =====================================================================
            %> @brief plots electrical potential
            % =====================================================================
            function plotPotential(obj)
                % plots electrical potential
                
                hold off
                rho = obj.rho{1} - obj.rho{2} + obj.rho{3};
                %rho = rho(round(obj.ny/2),round(obj.nx/2),:);
                rho = sum(sum(rho,1),2);
                rho = reshape(rho,length(rho),1);
                phi = obj.phi(round(obj.ny/2),round(obj.nx/2),:);
                phi = reshape(phi,length(phi),1);
                EN = obj.E_z(round(obj.ny/2),round(obj.nx/2),:);
                EN = 1e21/obj.N*reshape(EN,length(EN),1);
                
                subplot(3,1,1)
                plot(1e3*obj.z,rho)
                grid on
                xlabel('z [mm]')
                ylabel('\rho(z) [1/m^3]')
                subplot(3,1,2)
                plot(1e3*obj.z,phi)
                grid on
                xlabel('z [mm]')
                ylabel('U [V]')
                subplot(3,1,3)
                plot(1e3*obj.z,EN)
                grid on
                xlabel('z [mm]')
                ylabel('E/N [Td]')
                drawnow
                
            end
            
            % =====================================================================
            %> @brief collects and saves results
            % =====================================================================
            function results = getResults(obj)
                % collects and saves results
                
                results.EN = obj.EN;
                results.gas = obj.gas;
                results.mix = obj.mix;
                results.dir = obj.Xsec.dir;
                
                results.E.E_mean = obj.E.E_mean;
                
                results.E.energy = obj.E.energy;
                results.E.EEDF = obj.E.EEDF;
                results.E.EEPF = obj.E.EEPF;
                
                
                results.bulk.w = obj.bulk.w;
                results.bulk.w_err = obj.bulk.w_err;
                
                results.flux.w = obj.flux.w;
                
                results.bulk.DN = obj.bulk.DN;
                results.bulk.DN_err = obj.bulk.DN_err;
                
                results.flux.DN = obj.flux.DN;
                
                results.rates.count.eff = obj.rates.count.eff;
                results.rates.count.eff_err = obj.rates.count.eff_err;
                results.rates.conv.eff = obj.rates.conv.eff;
                
                results.rates.count.ion_tot = obj.rates.count.ion_tot;
                results.rates.count.ion_tot_err = obj.rates.count.ion_tot_err;
                results.rates.conv.ion_tot = obj.rates.conv.ion_tot;
                
                results.rates.count.att_tot = obj.rates.count.att_tot;
                results.rates.count.att_tot_err = obj.rates.count.att_tot_err;
                results.rates.conv.att_tot = obj.rates.conv.att_tot;
                
                results.elapsedTime = obj.t;
                
                results.ioncoeff = obj.a_directL/obj.mean.position(3,:);
                results.rates.direct = obj.a_directL/obj.t;
                

                results.particlenumber = obj.mean.particles{1}(end);

                results.particlenumberN = obj.N;

                results.a_eff = (obj.rates.count.eff*obj.N)/obj.flux.w(3);

                results.a_eff_from_fit = obj.fittingnu/obj.flux.w(3);

                results.a_eff_from_fit_diff = results.a_eff_from_fit*(1-(results.a_eff_from_fit*obj.bulk.DN(3))/(obj.N*obj.flux.w(3)))

                results.ratio_Townsend = results.ioncoeff/results.a_eff;

                results.a_eff_diff = results.a_eff*(1-(results.a_eff*obj.bulk.DN(3))/(obj.N*obj.flux.w(3)));

                results.ratio_Townsend_diff = results.ioncoeff/results.a_eff_diff;

                results.ratio_Townsend_fit_diff = results.ioncoeff/results.a_eff_from_fit_diff;

                results.ratio_rates_count = results.rates.direct/obj.fittingnu;

                %% save results:
                save('results','results')
                
            end
            
            
            
        end
        
    end
    
    
