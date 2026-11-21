
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

int main() {

    const int N = 5;
    const double EPS = 1e-6;

    double A[N][N] = {
        {20,20,20,20,20},
        {20,30,40,30,20},
        {20,40,100,40,20},
        {20,30,40,30,20},
        {20,20,20,20,20}
    };

    double B[N][N];

    int iterations = 0;
    double residual;

    do {

        residual = 0.0;

        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {

                if(i == 0 || j == 0 || i == N-1 || j == N-1)
                    B[i][j] = A[i][j];

                else
                    B[i][j] = (
                        A[i-1][j] +
                        A[i+1][j] +
                        A[i][j-1] +
                        A[i][j+1]
                    ) / 4.0;

                // Calculate maximum residual
                residual = max(residual,
                               abs(B[i][j] - A[i][j]));
            }
        }

        // Swap buffers
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                swap(A[i][j], B[i][j]);

        iterations++;

    } while(residual >= EPS);

    cout << "Converged successfully!\n";
    cout << "Iterations: " << iterations << endl;
    cout << "Final residual: " << residual << endl;

    cout << "\nFinal temperature grid:\n";

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++)
            cout << A[i][j] << "\t";

        cout << endl;
    }

    return 0;
}
