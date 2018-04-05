#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 60;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
 	particles.resize(num_particles);	
 	for (int i = 0; i < num_particles; ++i) 
	{
		 particles[i].x = dist_x(gen);
		 particles[i].y = dist_y(gen);
		 particles[i].theta = dist_theta(gen);	 
		 particles[i].weight = 1;
 	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	default_random_engine gen;
	std::normal_distribution<double> gauss_x(0, std_pos[0]);
	std::normal_distribution<double> gauss_y(0, std_pos[1]);
	std::normal_distribution<double> gauss_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; ++i)
	{
	    if( fabs(yaw_rate) < 0.001)
	    {
	      particles[i].x += velocity * delta_t * cos(particles[i].theta);
	      particles[i].y += velocity * delta_t * sin(particles[i].theta);
	    } 
	    else
	    {

	      particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t)-sin(particles[i].theta));
	      particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta + yaw_rate*delta_t));
	      particles[i].theta += yaw_rate*delta_t;
	    }
	    particles[i].x += gauss_x(gen);
	    particles[i].y += gauss_y(gen);
	    particles[i].theta += gauss_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0;i<(observations).size();++i)
	{
		double dis = 10000000.0;
		for (int j=0;j<(predicted).size();j++)
		{
			double xx = dist(predicted[j].x , predicted[j].y , observations[i].x , observations[i].y);
			if( xx < dis)
			{
				observations[i].id = predicted[j].id;
				dis = xx;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	const double normalize = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
	const double sig_x = (2*std_landmark[0]*std_landmark[0]);
	const double sig_y = (2*std_landmark[1]*std_landmark[1]);
	for (int i = 0; i < num_particles; ++i){
		// calculating which landmarks are within the sensor range
		std::vector<LandmarkObs> inrange;
		for (int j=0;j<(map_landmarks.landmark_list).size();++j)
		{
			if (dist(map_landmarks.landmark_list[j].x_f , map_landmarks.landmark_list[j].y_f ,particles[i].x ,particles[i].y) < sensor_range)
			{
				inrange.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, 
								map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
			}
		}

		vector<LandmarkObs> obs;
		double co=cos(particles[i].theta);
    		double si=sin(particles[i].theta);
		for(int k=0;k<(observations).size();++k)
		{
			LandmarkObs temp;
			temp.x= particles[i].x + (co*observations[k].x) - (si*observations[k].y);
			temp.y= particles[i].y + (si*observations[k].x) + (co*observations[k].y);
			obs.push_back(temp);
		}

		dataAssociation(inrange , obs);

		double product = 1.0;
		for(int l=0;l<(obs).size();++l)
		{
			Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs[l].id-1);
			double fir = obs[l].x - landmark.x_f;
			double sec = obs[l].y - landmark.y_f;
			double x_ = (fir * fir)/sig_x;
			double y_ = (sec * sec)/sig_y;
			double weight = exp(-(x_ + y_)) / normalize;
			product *=  weight;
		}

	    particles[i].weight = product;
	    weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> dist(weights.begin(), weights.end());
	vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);
	for(int i=0; i<num_particles; i++)
	{
		int idx = dist(gen);
		resampled_particles[i] = particles[idx];
	}
	particles = resampled_particles;
	weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}
string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
