#include "map.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>


coordinates::coordinates()
{
    init_gen();
}

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
    this->read_from_file(filename);

}

coordinates::coordinates(mat coord)
{
    init_gen();

    this->_coord=coord;
    this->_ndim = coord.n_cols;
    this->_ncities = coord.n_rows;
}


void coordinates::initialize(string inputfile)
{
    ifstream read("INPUT/"+inputfile);
    if(!read.is_open())
    {
        cerr << "ERROR: cannot open file INPUT/"<<inputfile<<"\n";
        exit(1);
    }

    bool def1 = false, def2 = false;

    string property, path="circle";
    while(!read.eof())
    {
        read >> property;
    
        if(property == "NCITIES")
        {
            read >> _ncities;
            def1=true;

        }
        else if(property == "PATH")
        {
            read >> path;
            def1=true;
        }
        else if (property == "COORD_FILE") 
        {
            read >>path;
            def2=true;
            // this->read_from_file(path);
        }
    }
    if(def1 && def2)
    {
        cerr << "Error: Multiple definition of coodinates in file INPUT/"
             << inputfile << "\n";
        exit(1);
    }
    if(!(def1 || def2 ) )
    {
        cerr<<"Error: No definition of coordinates in input file INPUT/"
            <<inputfile<<"\n";
            exit(1);
    }
    if(def2)
    {
        this->read_from_file(path);
    }
    else {
        this->_coord = mat(_ncities, _ndim);
    

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

        this->save_coordinates(
            "coord_"+path+
                "_"+to_string(_ncities)+"cities.dat"
        );
    }

    return;

}


vec coordinates::get_coord(int n) 
{
    check_index(n-1);
    return this->_coord.row(n-1).as_col();
}

double coordinates::dist(int n, int m)
{

    vec dif = this->get_coord(n)-this->get_coord(m);
    return norm(dif );
}

double coordinates::dist2(int n, int m)
{
    vec dif = this->get_coord(n)-this->get_coord(m);
    return dot(dif, dif);
}

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

void coordinates::read_from_file(string filename)
{

    ifstream in(filename);
    if(!in.is_open())
    {
        cerr<<"ERROR: cannot open coordinates file "
            <<filename<<"\n";
        exit(1);
    }
    string line;

    /*IGNORE FIRST 2 LINES*/
    getline(in, line);
    getline(in, line);

    // count number of cities
    int counter = 0;
    while (getline(in, line)) {
        counter+=1;
    }

    //init matrix
    this->_coord = mat(counter, _ndim);
   
    //reset stream
    in.clear();
    in.seekg(0, std::ios::beg);

    getline(in, line); //ignore first 2 lines
    getline(in, line); 

    char sep;
    int cordindex;

    for(int i=0; i<counter; i++)
    {
        //read line
        getline(in, line);

        //put it in stringstream
        stringstream data(line);
        data >> cordindex >> sep; //ignore index
        for(int dim=0; dim < _ndim; dim++)
        {

            // getline(data,temp, ',');
            double as;
            data>>as >> sep;
            this->_coord(i, dim)=as;
        }
    }
    in.close();

    this->_ncities=counter;

}