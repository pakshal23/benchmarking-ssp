% SampleViaCDF.m -- Gio -- Aout 2016 -- And later, et plus...
%
% Fonction Sample = SampleViaCDF(Densite,Grille);
%
% Echantillonne (approximativement) un scalaire (Sample) sous une densité 
% donnée pas ses valeurs (Densite) sur un grille (Grille).
%

function Sample = SampleViaCDF(Densite,Grille)

% Quelques tests
	if any(Densite<0), error('==> Il y a des la densite negative...'),end
	if all(Densite==00), error('==> Densite identiquement nulle...'),end

% Triture un peu la grille de valeurs 
	Delta = Grille(2) - Grille(1);

% Normalisation de la densité et calcul de la répartition
	Repartition = cumsum(Densite);
	Repartition = Repartition/Repartition(end);
	
% Génération d'un échantillon unifiorme
	LeRand = rand;

% Selection du bon intervalle
	[ ~ , Index] =  find( LeRand < Repartition );
	Index = min(Index);
	
	% et traitement selon le cas
	if Index == 1
		Sample = Grille(Index) + rand * Delta/2;
	elseif Index == length(Repartition)
		Sample = Grille(Index) - rand * Delta/2;
	else
		Sample = Grille(Index) + (rand-0.5)*Delta;
	end

	
	
