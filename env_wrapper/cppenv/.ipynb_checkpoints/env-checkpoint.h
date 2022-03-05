#ifndef ENV_H_
#define ENV_H_

#include <random>
#include <vector>
#include <array>

using ControlInfo = std::vector<std::array<double, 2>>; // omega, acc
using Observation = std::array<double, 50>;


struct Point2D
{
    double x, y;
};

class UAVModel2D
{
public:
    UAVModel2D(double x, double y, double v, double w);
    void step(double ang, double acc);

    double x() const { return m_x; }
    double y() const { return m_y; }
    double v() const { return m_v; }
    double w() const { return m_w; }

private:
    double m_x;
    double m_y;
    double m_v;
    double m_w;
};

int get_version();


class ManyUavEnv
{
public:
    ManyUavEnv(int uav_cnt, int random_seed, bool uav_die=true);
    void reset();
    void step(const ControlInfo& control);
    std::vector<Observation> getObservations() const;
    std::vector<double> getRewards();

    std::vector<Point2D> getObstacles() const;
    std::vector<Point2D> getUavs() const;
    std::vector<bool> getCollision() const;
    Point2D getTarget() const;

    bool isDone() const;

    int getCollisionWithObs() const { return collision_with_obs; }
    int getCollisionWithUav() const { return collision_with_uav; }
    int getInTargetArea() const { return in_target_area; }
    int getSuccCnt() const { return succ_cnt; }
    std::vector<bool> getStatus() const { return m_die; }
    
    // stat
    double getRadius() const;
    int getTa();
    int getDie();
    // stat
private:
    int m_uav_cnt;
    std::default_random_engine m_rnd_engine;
    std::vector<UAVModel2D> m_uavs;
    std::vector<Point2D> m_prev_pos;
    std::vector<Point2D> m_next_pos;
    std::vector<Point2D> m_obstacles;
    Point2D m_target;
    Point2D m_hidden_position;

    std::vector<bool> m_die;
    std::vector<int> m_status;

    mutable std::vector<bool> m_collision;
    int m_steps;

    //Point2D TARGET{ 1000, 1750 };

    mutable int in_target_area = 0;
    mutable int collision_with_obs = 0;
    mutable int collision_with_uav = 0;
    mutable int succ_cnt = 0;
};


#endif // !ENV_H_
