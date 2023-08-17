#include <iostream>

class KalmanFilter1D {
private:
    double state;           // Estimated state (e.g., position)
    double covariance;      // State's covariance (uncertainty)
    double processNoise;    // Process noise (Q)
    double measurementNoise; // Measurement noise (R)

public:
    KalmanFilter1D(double initialState, double initialCovariance,
                   double processNoise, double measurementNoise)
        : state(initialState), covariance(initialCovariance),
          processNoise(processNoise), measurementNoise(measurementNoise) {}

    // Predict the next state
    void predict(double controlInput) {
        // In 1D, the state transition model can be a simple addition
        state += controlInput;
        covariance += processNoise;
    }

    // Update the state based on measurement
    void update(double measurement) {
        double kalmanGain = covariance / (covariance + measurementNoise);
        state = state + kalmanGain * (measurement - state);
        covariance = (1 - kalmanGain) * covariance;
    }

    // Retrieve the current state estimate
    double getState() {
        return state;
    }
};

int main() {
    // Initialize the Kalman filter
    KalmanFilter1D kf(0.0, 1.0, 0.1, 0.5);

    // Simulate some measurements
    double measurements[] = {0.1, 0.2, 0.35, 0.5, 0.7};

    for (int i = 0; i < 5; i++) {
        // Predict next state (assuming no control input, for simplicity)
        kf.predict(0);

        // Update with measurement
        kf.update(measurements[i]);

        std::cout << "Updated state: " << kf.getState() << std::endl;
    }

    return 0;
}
