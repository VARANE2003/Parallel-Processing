#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

inline double f(double x)
{
        return sin(cos(x));
}

// WolframAlpha: integral sin(cos(x)) from 0 to 1  = 0.738643
// 0.73864299803689018
// 0.7386429980368901838000902905852160417480209422447648518714116299

int main(int argc, char *argv[])
{

        long physicalCores = sysconf(_SC_NPROCESSORS_CONF);
        long logicalProcessors = sysconf(_SC_NPROCESSORS_ONLN);


        double a = 0.0;
        double b = 1.0;
        unsigned long n = 24e8;
        long seed = time(0);

        if (argc == 2) {
                seed = atol(argv[1]);
        }
        else if (argc == 3) {
                n = atol(argv[1]);
                seed = atol(argv[2]);
        }

        const double h = (b-a)/n;
        const double ref = 0.73864299803689018;
        double sum = 0;
        double t0, t1;

        t0 = omp_get_wtime();
        #pragma omp parallel
        {
                int id = omp_get_thread_num();

                unsigned short buffer[3];
                buffer[0] = seed;
                buffer[1] = seed+id;
                buffer[2] = id;

                unsigned long i;
                #pragma omp for reduction(+:sum)
                for(i = 0; i < n; i++)
                {
                        double xi;
                        xi = erand48(buffer);
                        sum+=f(xi);
                }

                if(id == 0)            //master thread
                {
                        printf("Number of threads: %d\n", omp_get_num_threads());
                        printf("Physical Cores: %ld\n", physicalCores);
                        printf("Logical Processors: %ld\n", logicalProcessors);

                }
        }
        sum*=h;
        t1 = omp_get_wtime();

        printf("Result=%.16f Error=%e Rel.Error=%e Time=%lf seconds\n", sum, fabs(sum-ref), fabs(sum-ref)/ref, t1-t0);
        return 0;
}
