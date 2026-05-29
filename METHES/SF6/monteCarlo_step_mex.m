function sig = monteCarlo_step_mex(sig)
%#codegen

    sig = sig.freeFlight();
    sig = sig.collectMeanData();
    sig = sig.energyData();
    sig = sig.fluxData();
    sig = sig.bulkData();
    sig = sig.rateDataCount();
    sig = sig.rateDataConv();

    sig = sig.collisionMatrix();
    sig = sig.elasticCollision();
    sig = sig.inelasticCollision();
    sig = sig.ionCollision();
    sig = sig.attachCollision();

    sig = sig.checkSST();
    sig = sig.endSimulation();

    % Occasional printing
    if sig.collisions / 1e6 >= sig.line
        sig = sig.printOnScreen();
        sig.line = sig.line + 1;
    end
end