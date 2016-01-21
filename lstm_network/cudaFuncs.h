__device__ CDOUBLE exp1(CDOUBLE z) {
    double x = real(z);
    double a0 = abs(z);
    CDOUBLE ce1, cr, ct0, kc, ct;

    if (a0 == 0.0)
        ce1 = CDOUBLE(1.0e300, 0.0);
    else if ((a0 < 10.0) || (x < 0.0 && a0 < 20.0)) {
        ce1 = CDOUBLE(1.0, 0.0);
        cr = CDOUBLE(1.0, 0.0);
        for (int k = 1; k <= 150; k++) {
            cr = -(cr * double(k) * z)/CDOUBLE((k + 1.0) * (k + 1.0), 0.0);
            ce1 = ce1 + cr;
            if (abs(cr) <= abs(ce1)*1.0e-15)
                break;
        }
        ce1 = CDOUBLE(-EUL,0.0)-log(z)+(z*ce1);
    } else {
        ct0 = CDOUBLE(0.0, 0.0);
        for (int k = 120; k >= 1; k--) {
            kc = CDOUBLE(k, 0.0);
            ct0 = kc/(CDOUBLE(1.0,0.0)+(kc/(z+ct0)));
        }
        ct = CDOUBLE(1.0, 0.0)/(z+ct0);
        ce1 = exp(-z)*ct;
        if (x <= 0.0 && imag(z) == 0.0)
            ce1 = ce1-CDOUBLE(0.0, -PI);
    }
    return ce1;
}