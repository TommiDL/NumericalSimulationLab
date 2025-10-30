#include "map.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>


coordinates::coordinates(int n, string path)
{
    init_gen();
    this->_coord = mat(n, _ndim);
    

      // convert to lower case
    std::transform(
        path.begin(), path.end(), 
        path.begin(), 
        ::toupper
    );


    if(path == "CIRCLE")
    {
        fill_circle();
    }
    else
    {    
        fill_square();
    }

    this->_ncities=n;
}



coordinates::coordinates(string filename)
{
    init_gen();

    ifstream in(filename);

    int counter = 0;
    string line;
    while (getline(in, line)) {
        counter+=1;
        //cout << line << endl;
    }

    this->_coord = mat(counter, _ndim);
    
    in.clear();
    in.seekg(0, std::ios::beg);

    for(int i=0; i<counter; i++)
    {
        for(int dim=0; dim < _ndim; dim++)
        {
            in >> this->_coord(i, dim);
        }
    }
    in.close();

    this->_ncities=counter;
}

coordinates::coordinates(mat coord)
{
    init_gen();

    this->_coord=coord;
    this->_ndim = coord.n_cols;
    this->_ncities = coord.n_rows;
}


/*
return the coordinates corresponding to as a vec
*/
vec coordinates::get_coord(int n) 
{
    check_index(n-1);
    return this->_coord.row(n-1).as_col();
}

/*
return the eucledian distance between the cities n and m
*/
double coordinates::dist(int n, int m)
{

    vec dif = this->get_coord(n)-this->get_coord(m);
    return norm(dif );
}

/*
return the quadratic eucledian distance between the cities n and m
*/
double coordinates::dist2(int n, int m)
{
    vec dif = this->get_coord(n)-this->get_coord(m);
    return dot(dif, dif);
}

/*
return the eucledian lenght of the ...
*/
double coordinates::L1(vector<int> cities)
{
    double cost = 0;
    for(int k=0; k<cities.size()-1; k++)
    {
        cost+= this->dist(cities[k], cities[k+1]);
    }

    cost+=this->dist(cities[cities.size()-1], cities[0]);

    return cost;
}

double coordinates::L2(vector<int> cities)
{
    double cost = 0;
    for(int k=0; k<cities.size()-1; k++)
    {
        cost+= this->dist2(
            cities[k], cities[k+1]
        );
    }
    cost+=this->dist2(
        cities[cities.size()-1], 
        cities[0]
    );

    return cost;
}

void coordinates::check_index(int n)
{
    if ((n>=_ncities) or (n<0)) {
        //cerr  << "ERROR: coordinates index "
        //<<n<<" outside range 0-"<< _ncities
        //<< endl;
        throw invalid_argument(
            "ERROR: coordinates index "
            +to_string(n) 
            + " outside range 0-" 
            + to_string(_ncities)
        );
    }
    return;
}


void coordinates::fill_square(double size)
{
    cout << "cities generated inside a square"<<endl;
    for (int k=0; k< this->_coord.n_rows; k++)
    {
        this->_coord(k, 0) = this->_rnd.Rannyu(0, size);
        this->_coord(k, 1) = this->_rnd.Rannyu(0, size);
    }
    return;
}

void coordinates::fill_circle(double ray)
{
    cout << "cities generated along a circle"<<endl;
    for(int k = 0; k < this->_coord.n_rows; k++)
    {
        double angle = _rnd.Rannyu(0, 2*M_PI);
        this->_coord(k, 0) = ray * cos(angle);
        this->_coord(k, 1) = ray * sin(angle);
    }
    return;
}

void coordinates::init_gen(string primesfile, string seedfile)
{
  // INIT RANDOM GENERATOR
  int p1, p2; // Read from ../INPUT/Primes a pair of numbers to be used to initialize the RNG
  ifstream Primes(primesfile);
  Primes >> p1 >> p2 ;
  Primes.close();
  int seed[4]; // Read the seed of the RNG
  ifstream Seed(seedfile);
  Seed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
  this->_rnd.SetRandom(seed,p1,p2);


}

void coordinates::print()
{
    cout << "number of cities: "<<_ncities<<endl;
    for(int k=0; k<_ncities; k++)
    {
        cout <<k<<")  ";
        for(int dim=0; dim<_ndim; dim++) 
            cout<<setw(16)<<this->_coord(k, dim);
        cout<<endl;
    }
}

void coordinates::save_coordinates(string filename)
{

    ofstream out(filename);

    out << "#number of cities: "<<_ncities<<endl;
    out << "number,x,y" <<endl;
    for(int k=0; k<_ncities; k++)
    {
        out <<k;
        for(int dim=0; dim<_ndim; dim++) 
            out<<","<<this->_coord(k, dim);
        out<<endl;
    }
}

