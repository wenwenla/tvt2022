#include "env.h"
#include <cmath>
#include <algorithm>
#include <iostream>


constexpr auto PI = (3.14159265358979323846);
constexpr auto EPS = (1e-8);
constexpr auto OBSTACLE_CNT = (0);
constexpr auto COLLISION_R = (30);
constexpr auto TARGET_R = (100);
constexpr auto UAV_COLLISION_R = (3);
constexpr auto DETECTED_RANGE = (200);
constexpr auto UAV_UAV_COMM = (50);


constexpr auto DIE_PENALTY = (100); // 10000 (eval)  100 (train)
constexpr auto MAX_DIST_PENALTY = (50);
constexpr auto MAX_DIST_RANGE = (500);
constexpr auto TASK_REW = (20);  // 20000 (eval)  20 (train)


int get_version() {
    return 5;
}

inline double l2norm(double x, double y)
{
    return sqrt(x * x + y * y);
}

inline double cross(double lx, double ly, double rx, double ry)
{
    return lx * ry - rx * ly;
}

inline double dot(double lx, double ly, double rx, double ry)
{
    return lx * rx + ly * ry;
}

UAVModel2D::UAVModel2D(double x, double y, double v, double w) : m_x(x), m_y(y), m_v(v), m_w(w)
{
}

void UAVModel2D::step(double ang, double acc)
{
    m_w += ang;
    m_v += acc;
    if (m_v > 5) m_v = 5;
    if (m_v < 0) m_v = 0;

    while (m_w > PI) {
        m_w -= PI * 2;
    }
    while (m_w < -PI) {
        m_w += PI * 2;
    }
    m_x += m_v * cos(m_w);
    m_y += m_v * sin(m_w);
}

ManyUavEnv::ManyUavEnv(int uav_cnt, int random_seed, bool uav_die) :
        m_uav_cnt(uav_cnt),
        m_rnd_engine(random_seed),
        m_target{},
        m_steps(0)

{
    reset();
    m_collision.resize(OBSTACLE_CNT);
}

void ManyUavEnv::reset()
{
    m_uavs.clear();
    m_obstacles.clear();
    std::uniform_real_distribution<double> uav_dist_x(0, 2000), uav_dist_y(0, 400);
    for (int i = 0; i < m_uav_cnt; ++i) {
        m_uavs.emplace_back(
                uav_dist_x(m_rnd_engine),
                uav_dist_y(m_rnd_engine),
                0.0,
                PI / 2
        );
    }
    std::uniform_real_distribution<double> obs_dist_x(0, 2000), obs_dist_y(500, 1500);
    for (int i = 0; i < OBSTACLE_CNT; ++i) {
        m_obstacles.push_back({
                                      obs_dist_x(m_rnd_engine),
                                      obs_dist_y(m_rnd_engine)
                              });
    }
    // initialize target position
    std::uniform_int_distribution<int> target_dist_x(500, 1500), target_dist_y(1600, 1800);
    m_target.x = target_dist_x(m_rnd_engine);
    m_target.y = target_dist_y(m_rnd_engine);
    m_steps = 0;

    m_die.clear();
    m_status.clear();
    for (int i = 0; i < m_uav_cnt; ++i) {
        m_die.push_back(false);
        m_status.push_back(0);
    }
    succ_cnt = 0;
}

void ManyUavEnv::step(const ControlInfo& control)
{
    in_target_area = 0;
    collision_with_obs = 0;
    collision_with_uav = 0;

    m_prev_pos.clear();
    m_next_pos.clear();
    m_prev_pos.resize(m_uav_cnt);
    m_next_pos.resize(m_uav_cnt);
#pragma omp parallel for default(none) shared(control)
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_die[i]) continue;
        m_prev_pos[i] = { m_uavs[i].x(), m_uavs[i].y() };
        m_uavs[i].step(control[i][0], control[i][1]);
        m_next_pos[i] = { m_uavs[i].x(), m_uavs[i].y() };
    }
    m_steps += 1;
}

std::vector<Observation> ManyUavEnv::getObservations() const
{
    std::vector<Observation> result(m_uav_cnt);

    Point2D center {0, 0};

    int cnt = 0;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_status[i] != 1) {
            center.x += m_uavs[i].x();
            center.y += m_uavs[i].y();
            cnt += 1;
        }
    }
    center.x /= cnt;
    center.y /= cnt;

    #pragma omp parallel for default(none) shared(result, center)
    for (int i = 0; i < m_uav_cnt; ++i) {
        Observation obs;

        obs[0] = (m_uavs[i].x() - 1000.) / 2000.;
        obs[1] = (m_uavs[i].y() - 1000.) / 2000.;
        obs[2] = (m_uavs[i].v() - 2.5) / 5.;
        obs[3] = (m_uavs[i].w()) / PI;

        std::vector<std::pair<int, int>> index_dist;
        for (int j = 0; j < m_uav_cnt; ++j) {
            if (j == i) continue;
            if (m_die[j]) continue;
            double dist = l2norm(m_uavs[j].x() - m_uavs[i].x(), m_uavs[j].y() - m_uavs[i].y());
            index_dist.emplace_back(dist, j);
        }
        std::sort(index_dist.begin(), index_dist.end());

        // 6 + 4 * 6
        for (int j = 0; j < 10; ++j) {
            if (j < index_dist.size() && index_dist[j].first < UAV_UAV_COMM) {
                obs[4 + j * 4] = (m_uavs[index_dist[j].second].x() - 1000.) / 2000.;
                obs[5 + j * 4] = (m_uavs[index_dist[j].second].y() - 1000.) / 2000.;
                obs[6 + j * 4] = (m_uavs[index_dist[j].second].v() - 2.5) / 5.;
                obs[7 + j * 4] = (m_uavs[index_dist[j].second].w()) / PI;
            } else {
                obs[4 + j * 4] = -1.0;
                obs[5 + j * 4] = -1.0;
                obs[6 + j * 4] = 0.0;
                obs[7 + j * 4] = 0.0;
            }
        }
        // generate target info
        obs[44] = (m_target.x - 1000.) / 500.;
        obs[45] = (m_target.y - 1700.) / 100.;
        obs[46] = int(m_die[i]);

        obs[47] = m_steps / 500.;
        obs[48] = (center.x - 1000.) / 2000.;
        obs[49] = (center.y - 1000.) / 2000.;

        result[i] = obs;
    }
    return result;
}

std::vector<double> ManyUavEnv::getRewards()
{
    std::fill(m_collision.begin(), m_collision.end(), false);

    std::vector<double> result(m_uav_cnt, 0.0);
    Point2D center {0, 0};

    int cnt = 0;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_status[i] != 1) {
            center.x += m_uavs[i].x();
            center.y += m_uavs[i].y();
            cnt += 1;
        }
    }
    center.x /= cnt;
    center.y /= cnt;

    std::vector<bool> die_on_this_step(m_uav_cnt, false);

    #pragma omp parallel for default(none) shared(result, center, die_on_this_step)
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_die[i]) {
            result[i] = 0;
            continue;
        }

        //distance based rule
        double dp = l2norm(m_prev_pos[i].x - m_target.x, m_prev_pos[i].y - m_target.y);
        double dn = l2norm(m_next_pos[i].x - m_target.x, m_next_pos[i].y - m_target.y);
        result[i] += dp - dn; // original distance


        // check uav & uav collision
        bool collision = false;
        for (int j = 0; j < m_uav_cnt; ++j) {
            if (m_die[j] || i == j) continue;
            if (l2norm(m_next_pos[j].x - m_next_pos[i].x, m_next_pos[j].y - m_next_pos[i].y) < UAV_COLLISION_R) {
                collision = true;
                die_on_this_step[i] = true;
                collision_with_uav += 1;
                break;
            }
        }
        if (collision) {
            result[i] -= DIE_PENALTY;
            m_status[i] = 1;
        }

        for (int j = 0; j < m_uav_cnt; ++j) {
            if (m_die[j]) continue;
            if (i != j) {
                double dist = l2norm(m_next_pos[j].x - m_next_pos[i].x, m_next_pos[j].y - m_next_pos[i].y);
                if (dist < 20 && dist > 3) {
                    double k = -20.0 / 17.0;
                    double p = k * (dist - 20);
                    result[i] -= p;
                }
            }
        }

        // dist to center
        double dtc = l2norm(m_next_pos[i].x - center.x, m_next_pos[i].y - center.y);
        double lim_R = UAV_COLLISION_R * sqrt(2 * m_uav_cnt) * 1.5;

        if (dtc < lim_R) {
            result[i] += 5.0;
        } else if (dtc > MAX_DIST_RANGE) {
            result[i] -= MAX_DIST_PENALTY;
        } else {
            double k = (MAX_DIST_PENALTY) / (MAX_DIST_RANGE - lim_R);
            result[i] -= k * (dtc - lim_R);
        }


        // in the circle
        if (l2norm(m_next_pos[i].x - m_target.x, m_next_pos[i].y - m_target.y) < TARGET_R) {
            result[i] += TASK_REW;
            in_target_area += 1;
            die_on_this_step[i] = true;
            m_status[i] = 2;
            if (!m_die[i]) {
                succ_cnt += 1;
            }
        }
    }

    #pragma omp parallel for default(none) shared(result)
    for (int i = 0; i < m_uav_cnt; ++i) {
        result[i] /= 100.0;
    }

    #pragma omp parallel for default(none) shared(result, die_on_this_step)
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (die_on_this_step[i]) {
            m_die[i] = true;
        }
    }
    return result;
}

std::vector<Point2D> ManyUavEnv::getObstacles() const
{
    return m_obstacles;
}

std::vector<Point2D> ManyUavEnv::getUavs() const
{
    return m_next_pos;
}

std::vector<bool> ManyUavEnv::getCollision() const
{
    return m_collision;
}

Point2D ManyUavEnv::getTarget() const
{
    return m_target;
}

bool ManyUavEnv::isDone() const
{
    bool all_die = true;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (!m_die[i]) all_die = false;
    }
    return m_steps == 500 || all_die;
}

double ManyUavEnv::getRadius() const {
    Point2D center{0, 0};
    int tot = 0;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_status[i] != 1) {
            center.x += m_uavs[i].x();
            center.y += m_uavs[i].y();
            tot += 1;
        }
    }
    center.x /= tot;
    center.y /= tot;
    double tot_dist = 0;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (!m_die[i]) {
            tot_dist += l2norm(m_uavs[i].x() - center.x, m_uavs[i].y() - center.y);
        }
    }
    return (tot == 0) ? 0 : tot_dist / tot;
}

int ManyUavEnv::getTa() {
    int cnt = 0;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_status[i] == 2) ++cnt;
    }
    return cnt;
}

int ManyUavEnv::getDie() {
    int cnt = 0;
    for (int i = 0; i < m_uav_cnt; ++i) {
        if (m_status[i] == 1) ++cnt;
    }
    return cnt;
}
