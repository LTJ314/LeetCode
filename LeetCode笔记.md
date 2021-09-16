# LeetCode笔记

## 二分查找

[详解二分查找算法 - murphy_gb - 博客园 (cnblogs.com)](https://www.cnblogs.com/kyoner/p/11080078.html)

二分查找的关键在于对搜索区间的理解

### 33.搜索旋转排序数组

[33. 搜索旋转排序数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

有序数组只经过了一次旋转，因此将数组从中间分开，必定有一部分是有序的，另一部分可能是无序的

如果nums[0]<=nums[mid]，则有序的部分是0~mid，此时判断target的大小，如果target的大小在nums[0]和nums[mid]之间，则将right更新为mid-1，继续在left~right之间寻找target，否则将left更新为mid+1，说明target大于nums[mid]及其左边的数

否则有序的部分是mid+1~n-1，如果target的大小在nums[mid]和nums[n-1]之间，则将left更新为mid+1，否则将right更新为mid-1。

直到nums[mid]=target或者数组被搜索完（left>right）为止



### 34.在排序数组中查找元素的第一个和最后一个位置

[34. 在排序数组中查找元素的第一个和最后一个位置 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

二分查找获取左侧边界和右侧边界

### 162.寻找峰值

[162. 寻找峰值 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-peak-element/)

### 287.寻找重复数

[287. 寻找重复数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-the-duplicate-number/)

重复数re的范围是1~n，整个数组中不大于re的数的个数必定大于re个，

如果一个数x小于re，则数组中不大于x的数量必定不超过x个

如果一个数x大于re，则数组中不大于x的数量必定大于x个

根据这一性质，使用二分法查找重复数

每次二分遍历数组计算不大于mid的数的数量count，通过二分缩小mid的范围

如果count<=mid，则说明重复数大于mid，left要增大到mid+1

如果count>mid，则说明重复数小于或等于mid，right减小到mid-1，符合这一情况的最小的mid就是重复数，也就是当left=right二分结束时，满足该条件的mid最小

### 436.寻找右区间

[436. 寻找右区间 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-right-interval/)

排序+二分查找

首先使用哈希表map存储intervals数组中starti 和下标 i 的关系，方便第二步的排序

然后对intervals数组排序，按照`intervals[i][0]`的升序排序

遍历数组，对每一个intervals[i]的endi作为target，在有序数组intervals的start中二分查找target的左侧边界

需要注意的是最终结果的下标要通过哈希表获取

### 704.二分查找

[704. 二分查找 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/binary-search/)

最基础的二分查找

## 快慢指针

### 141.环形链表

[141. 环形链表 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/linked-list-cycle/)

定义两个指针，一快一慢。慢指针每次只移动一步，而快指针每次移动两步。初始时，慢指针在位置 head，而快指针在位置 head.next。这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表。否则快指针将到达链表尾部，该链表不为环形链表。

注意本题中不能把slow.val==fast.val作为判断有环的条件，而是slow==fast，因为链表中节点的值可能相同

### 142.环形链表II

[142. 环形链表 II - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

判断是否有环与上题相同，但关键在于找到环的入口节点

判断是否有环，slow初始值为head，fast的初始值为head或者head.next都可以

但是本题要找到环的入口，如果fast的初始值为head.next，在找环入口的过程中可能出现死循环的情况，fast的初始值只能是head。

下面通过数学推导说明原理

slow和fast最开始都在head，slow和fast在环内相遇

如下图所示，设链表中环外部分的长度为 a。slow 指针进入环后，又走了 b 的距离与fast 相遇。此时，fast 指针已经走完了环的 n 圈，因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nc。

![fig1](https://assets.leetcode-cn.com/solution-static/142/142_fig1.png)

fast 指针走过的距离都为 slow 指针的 2 倍。因此，有
a+(n+1)b+nc=2(a+b)⟹a=c+(n−1)(b+c)

有了 a=c+(n-1)(b+c)的等量关系，我们会发现：从相遇点到入环点的距离加上 n−1 圈的环长，恰好等于从链表头部到入环点的距离。

因此，当发现slow 与 fast 相遇时，head指向链表头部；随后，head和slow 每次向后移动一个位置。最终，它们会在入环点相遇。



### 287.寻找重复数

[287. 寻找重复数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-the-duplicate-number/)

由于本题的数组的长度为n+1，数组元素范围都是1~n，因此可以使用下标和数组元素的映射关系

如果数组中没有重复的数，以数组 [1,3,4,2]为例，我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)f(n)，
其映射关系 n->f(n)为：
0->1
1->3
2->4
3->2
我们从下标为 0 出发，根据 f(n)f(n) 计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样可以产生一个类似链表一样的序列。
0->1->3->2->4->null

如果数组中有重复的数，以数组 [1,3,4,2,2] 为例,我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)f(n)，
其映射关系 n->f(n) 为：
0->1
1->3
2->4
3->2
4->2
同样的，我们从下标为 0 出发，根据 f(n)f(n) 计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推产生一个类似链表一样的序列。
0->1->3->2->4->2->4->2->……
这里 2->4 是一个循环，那么这个链表可以抽象为下图：

![287.png](https://pic.leetcode-cn.com/999e055b41e499d9ac704abada4a1b8e6697374fdfedc17d06b0e8aa10a8f8f6-287.png)


从理论上讲，数组中如果有重复的数，那么就会产生多对一的映射，这样，形成的链表就一定会有环路了，

综上
1.数组中有一个重复的整数 <==> 链表中存在环
2.找到数组中的重复整数 <==> 找到链表的环入口

至此，问题转换为 142 题。那么针对此题，快、慢指针该如何走呢。根据上述数组转链表的映射关系，可推出
142 题中慢指针走一步 slow = slow.next ==> 本题 slow = nums[slow]
142 题中快指针走两步 fast = fast.next.next ==> 本题 fast = nums[nums[fast]]

第一步通过快慢指针找到环内的元素

第二步从数组头开始找到环的入口元素，即重复数

具体代码如下：

```java
	public int findDuplicate(int[] nums) {
        int slow=nums[0];
        int fast=nums[nums[0]];
        while(slow!=fast)
        {
            slow=nums[slow];
            fast=nums[nums[fast]];
        }
        //此时slow在环内，但是slow不一定就是重复数，即环的入口
        int p=0;
        while(p!=slow)
        {
            p=nums[p];
            slow=nums[slow];
        }
        return slow;
    }
```



## 双指针

### 524.通过删除字母匹配到字典里最长单词

[524. 通过删除字母匹配到字典里最长单词 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

遍历字典，将字典中的字符串cur与字符串s匹配，使用双指针匹配

匹配过程如下：

```java
			int curlen=cur.length();
            int index=0;
            int j=0;
            while(j<curlen&&index<m)
            {
                while(index<m&&s.charAt(index)!=cur.charAt(j))
                {
                    index++;
                }
                if(index<m)
                {
                    j++;
                    index++;
                }
            }
```

如果 j 最后达到了curlen，则说明s能够通过删除得到字符串cur，

将cur与之前暂存结果的字符串re比较（re的初始值为空字符串）

如果cur更长，则将re更新为cur

如果cur和re长度相同，则比较它们的字典序，字典序更小的就是新的re

否则re不变

```java
			if(j==curlen)
            {
                if(relen<curlen)
                {
                    re=cur;
                }
                else if(relen==curlen&&re.compareTo(cur)>0)//字典序最小
                {
                    re=cur;
                }
            }
```

改进方法：

首先将字典排序，按照长度从大到小，字典序从小到大的顺序

```java
		Collections.sort(dictionary,new Comparator<String>() {
            public int compare(String word1, String word2) {
                if (word1.length() != word2.length()) {
                    return word2.length() - word1.length();
                } else {
                    return word1.compareTo(word2);
                }
            }
        });
```

然后遍历字典，匹配过程与之前相同，由于字典是经过排序的，所以第一个能匹配的字符串就是最长且字典序最小的字符串，可以直接返回当作结果

### 581.最短无序连续子数组

[581. 最短无序连续子数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

无序子数组满足，从左向右找到第一个nums[i]>nums[i+1]，从右向左找到第一个nums[j]<nums[j-1]，即确定左右两边不满足升序的元素位置，如果j>i，说明现在的数组就是升序的，不没有无序子数组，即结果为0

但是现在的子数组nums{i,...,j}并不一定就是最短无序连续子数组，因为i到j之间仍可能出现无序，比如 i 到 j 之间的元素可能比 i 左边的元素更小或者比 j 右边的元素更大，这个时候仅仅将 i 到 j 之间的元素升序，是不能使得整个nums数组升序的

所以继续遍历找到从i到j之间的最小值min和最大值max，在 i 的左边找到第一个小于min的元素下标，在 j 的右边找到第一个大于max的元素下标，即可确定最短无序连续子数组

如果此时两个下标相等，则数组是升序的，结果为0

否则结果为(j-1)-(i+1)+1，i,j分别为更新后的元素下标

### 881.救生艇

[881. 救生艇 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/boats-to-save-people/)

关键信息：每艘船最多只能载两个人

要使船的数量最小，那么尽可能多的载两个人，

采用贪心策略，最理想的情况是：最重和最轻的人坐在一起，次重和次轻的人坐在一起，......

然而如果较重的人已经超过限制的重量，那么只能单独坐一艘船

因此采用排序+双指针的方式，首先对体重数组people[n]排序，left最轻，right最重，如果两者之和不超过限制limit，则可同坐一艘船，两个指针同时向中间移动，否则较重的right坐一艘船，right左移，left不移动进入与right-1求和的下一次判断

## 排序

### 归并排序

### 148.排序链表

[148. 排序链表 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/sort-list/)

递归法：

通过递归实现链表归并排序

分割链表，找到当前链表中点mid，从中点将链表断开分为left和right两部分，直到链表被分割为一个个单独的节点

合并链表，将两个排序链表合并为一个排序链表

由于第一步已经将链表分割为单独的节点，因此合并部分也是首先将单独的节点两两合并，然后依此类推，直至完整的链表被合并完毕

找链表中点采用快慢指针的方法：

```java
		ListNode mid=head;
        ListNode tail=head;
        ListNode pre=mid;
        while(tail!=null&&tail.next!=null)//找到链表的中点  
        {
            pre=mid;
            mid=mid.next;
            tail=tail.next.next;
        }
```

在中点处将链表分为两段：

```java
//将链表分为两段
        pre.next=null;
        ListNode left=sortList(head);
        ListNode right=sortList(mid);
//递归之后，链表将会被分为一个个节点，并且使用归并排序对这些节点整合为结果链表
```

对有序链表做合并：

```java
	private ListNode merge(ListNode a,ListNode b)
    {
        ListNode newhead=new ListNode(0);
        ListNode p=newhead;
        while(a!=null&&b!=null)
        {
            if(a.val>b.val)
            {
                p.next=b;
                b=b.next;
            }
            else
            {
                p.next=a;
                a=a.next;
            }
            p=p.next;
        }
        if(a!=null)
        {
            p.next=a;
        }
        else{
            p.next=b;
        }
        return newhead.next;
    }
```



### 快速排序

快速排序原理：[快速排序 | 菜鸟教程 (runoob.com)](https://www.runoob.com/w3cnote/quick-sort.html)

### 215.数组中的第K个最大元素

[215. 数组中的第K个最大元素 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

暴力：

全排序，从小到大的顺序，取下标为n-k的元素

堆：

利用小根堆，维护一个长度为k的优先队列，队列从头到尾是递增的顺序，首先将数组的前K个数入队，再加入数组后面的数num时，如果队头的数小于num，则队头出队，num入队，遍历完整个数组之后，队头的数就是数组中第K大的元素

快速排序：

利用快速排序的思想，一轮快排之后，基准元素x左边都是小于x的数，右边都是大于x的数，即基准元素位于它应该在的位置。

即选择到基准元素下标为n-k

### 面试题17.14.最小K个数

[面试题 17.14. 最小K个数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/smallest-k-lcci/)

找出数组中最小的k个数

最先想到将数组全排序，按照从小到大的顺序，取前k个数组成结果数组

利用堆的性质：

小根堆，使用PriorityQueue，队列从头到尾是递增的顺序，然后截取队头的前k个数

大根堆，也是使用优先队列，队列从头到尾是递减的顺序，维护一个长度为k的优先队列，新加入的数如果小于队头，则队头出队，较小的那个数入队，这样遍历完整个arr数组之后，队列中就是前K个最小的数

快速排序：

利用快速排序的思想，一轮快排之后，基准元素x左边都是小于x的数，右边都是大于x的数

由于本题只需要最小的k个数，因此只需要维护基准元素左边的数

如果选择的基准元素经过一轮快排后，下标cur等于k，则此时基准元素左边就是最小的k个数

如果选择的基准元素经过一轮快排后，下标cur大于k，则满足结果的基准元素在左边，继续对左边元素(left,cur-1)进行快排

如果选择的基准元素经过一轮快排后，下标cur小于k，则满足结果的基准元素在右边，继续对右边元素(cur+1,right)进行快排

### 451.根据字符出现频率排序

[451. 根据字符出现频率排序 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/sort-characters-by-frequency/)

首先用Map<Character,Integer> map将字符串s中的字符和出现数量记录下来

再用List<Character> list将字符录入，并且重写其排序函数，根据map中字符数量从大到小排序，从而遍历list和在map中对应字符的数量就可以拼接得到结果

### 堆排序

### 502.IPO

[502. IPO - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/ipo/)

堆+贪心

要使得资本最大化，首先要保证每次投的项目成本capital小于当前资本w，其次是净利润profit尽可能的高，而且每次投项目后拥有的资本w要加上获得的净利润profit

首先对项目进行排序，按照成本从小到大的顺序排序，便于后续判断项目是否能投

然后进入选择k个项目的过程：

使用优先队列实现大根堆，队头的净利润最高

遍历排序后的项目数组，成本不超过当前资本w的净利润加入优先队列，直到成本超过当前资本的项目停止加入队列，到此为止，队列头部的净利润就是满足不超过w且利润最高的项目，将其加入w

遍历完所有项目后，得到可获得的最大资本



## 滑动窗口

### 1838.最高频元素的频数

[1838. 最高频元素的频数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element/)

排序+滑动窗口

要求元素的最高频数，首先对nums排序。维护nums下标的滑动窗口，left~right，表示能够经过不大于k次操作后使得窗口内对应的nums元素都是相同的，那么窗口长度的最大值就是题目要求的结果

left初始为0，right初始为1，结果re初始为1（频数最少为1）

那么窗口长度为right-left+1

right每次向右移动一位，那么对应的所需要的+1操作次数total就需要增加(right-left)*(nums[right]-nums[right-1])次，+1操作是为了使得nums[right]左边的元素(left~right-1)增大到与nums[right]相同

total需要满足不大于k，如果total大于k，需要将left右移，从而减小total的值

left每向右移动一位，所需要的+1操作次数total减少(nums[right]-nums[left])次

符合不大于k的total最大时，就是当前right对应的窗口最长的情况，从而re在right从1到n的取值中取最大的

## 拒绝采样

### 470.用Rand7()实现Rand10()

[470. 用 Rand7() 实现 Rand10() - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

本题的关键在于对概率的理解

随机生成1到10之间的整数，生成每个整数的概率为1/10

利用rand7()生成1到6之间的整数a，即当a=7时，拒绝a为7的情况，继续调用rand7()生成新的a，直至a不等于7为止，a为奇或偶的概率为1/2

利用rand7()生成1到5之间的整数b，同样是拒绝b>5的情况，那么b为1到5之间的数的概率都是1/5

从而将1到10拆解为1到5和6到10两段，随机生成的整数落在这两段的概率理应都为1/2，这一点通过a的奇偶来体现

如果a为偶数，则结果是1到5之间，即b

如果a为奇数，则结果是6到10之间，即b+5

这样将1/10的概率拆解为1/2*1/5

具体代码如下：

```java
public int rand10() {
        int a=rand7();
        int b=rand7();
        while(a==7)
        {
            a=rand7();
        }
        while(b>5)
        {
            b=rand7();
        }
        if(a%2==0)
        {
            return b;
        }
        else
        {
            return b+5;
        }
    }
}
```



## 前缀和

### 238.除自身以外数组的乘积

[238. 除自身以外数组的乘积 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/product-of-array-except-self/)

首先想到求出所有元素的乘积，并且判断数组中0的个数，满足O(n)的时间复杂度，但是需要用到除法

我们不必将所有数字的乘积除以给定索引处的数字得到相应的答案，而是利用索引左侧所有数字的乘积和右侧所有数字的乘积（即前缀与后缀）相乘得到答案。

对于给定索引 i，我们将使用它左边所有数字的乘积乘以右边所有数字的乘积。

尽管上面的方法已经能够很好的解决这个问题，但是空间复杂度并不为常数。

由于输出数组不算在空间复杂度内，那么我们可以将 L 或 R 数组用输出数组来计算。先把输出数组当作 L 数组来计算，然后再动态构造 R 数组得到结果。

### 523.连续的子数组和

[523. 连续的子数组和 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/continuous-subarray-sum/)

连续子数组和可由前缀和之差表示

符合是k的倍数的前缀和之差应满足，两个前缀和模k的结果相同

因此利用哈希表存放前缀和模k，和对应的下标

哈希表还需put(0,-1)，考虑前缀和直接就是k的倍数的情况

遍历nums是否存在相同模k结果且符合题目条件下标之差>=2的前缀和

### 525.连续数组

[525. 连续数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/contiguous-array/)

题目要求连续子数组中含有相同数量的0和1，相当于将0替换为-1后，连续子数组之和为0

同样采用前缀和之差表示连续子数组之和，即两个前缀和相等就符合题设

哈希表存放前缀和与对应下标

哈希表还需put(0,-1)，考虑前缀和直接就是0的情况

遍历nums是否存在前缀和相等，两个前缀和之差就是连续子数组的长度

### 1744.你能在你最喜欢的那天吃到你最喜欢的糖果吗

[1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？ - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/)

计算candiesCount的前缀和sum[]

划分区间

在favoriteDayi天所吃糖果数量区间为[favoriteDayi,favoriteDayi*dailyCapi]，最少就是每天一颗，最多就是每天dayliCapi颗

要在规定的天数吃到favoriteTypei的糖果，需要吃的糖果数量区间为`[sum[favoriteTypei-1]+1,sum[favoriteTypei]]`，考虑左边界情况

这两个区间有交集，answer就为True

## 动态规划

![转化过程](https://pic.leetcode-cn.com/2cdf411d73e7f4990c63c9ff69847c146311689ebc286d3eae715fa5c53483cf-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-03-08%2010.23.03.png)

### 5.最长回文子串

[5. 最长回文子串 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-palindromic-substring/)

找字符串s中最长的回文子串

​	回文子串可按字符串的长度分类：

（1）只有一个字符时，必是回文子串；

（2）有两个字符时，这两个字符相等就是回文子串；

（3）大于两个字符时，要求首尾的字符相等，且删除首尾字符后剩下的字符串仍为回文子串。

取boolean类型的`dp[i][j]`表示下标从 i 到 j 的子串是否为回文子串，i<=j

根据回文子串的分类可以得到状态转移方程为：
$$
dp[i][j] = \begin{cases}
true &\text{i=j}\\
true &\text{j-i=1且s(i)=s(j)}\\
true &\text{j-i>1且s(i)=s(j)且dp[i+1][j-1]=true}\\
false &\text{其它情况}\\
\end{cases}
$$
由于`dp[i][j]`需要由`dp[i+1][j-1]`得到，所以 i 的遍历顺序是从大到小，而 j 的遍历顺序是从小到大

题目要求的是最长的回文子串，因此，使用begin和end表示最长回文子串的首尾字符下标，当`dp[i][j]=true`时，begin和end取 j-i 最大的 i 和 j ，最后的题解取相应下标的子串

```java
        for(int i=n-1;i>=0;i--)
        {
            for(int j=i;j<n;j++)
            {
                if(j==i)//只有一个字符
                {
                    dp[i][j]=true;
                }
                else if(i==j-1&&s.charAt(i)==s.charAt(j))//两个字符相等
                {
                    dp[i][j]=true;
                }
                else if(j-i>1&&dp[i+1][j-1]&&s.charAt(i)==s.charAt(j))//大于两个字符
                {
                    dp[i][j]=true;
                }
                if(dp[i][j]&&j-i>end-begin)
                {
                    begin=i;
                    end=j;
                }
            }
        }
```



### 96.不同的二叉搜索树

[96. 不同的二叉搜索树 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/unique-binary-search-trees/)

n个节点的二叉搜索树

计算每个节点作为根节点时的情况，将其求和即为答案

当一个节点作为根节点时，所有的情况数等于左子树种数*右子树种数

因此可以得到状态转移方程

### 152.乘积最大子数组

[152. 乘积最大子数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/maximum-product-subarray/)

对于乘积来说，如果当前为负数，那么最大值乘以它之后会变成最小值，而最小值会变成最大值，利用这个特性，维持max和min两个变量分别表示当前数组的最大和最小乘积

因此当nums[i]<0时，max[i-1]需要与min[i-1]互换

以第i个元素结尾的乘积最大子数组的乘积 max[i] 可以考虑将当前 nums[i] 乘以之前的 max[i-1]，或者单独就是 nums[i] ，在这两种情况中取最大的

同理，min[i]则是在两种情况中取最小的

题解的最大连续子数组的乘积需要从 max[i] 中找最大的

空间优化后

```java
for(int i=0;i<n;i++)
{
    if(nums[i]<0)
    {
                int temp=max[i-1];
                max[i-1]=min;
                min=temp;
    }
    max=Math.max(nums[i]*max,nums[i]);
    min=Math.min(nums[i]*min,nums[i]);
    re=Math.max(max,re);
}
```



### 221.最大正方形

[221. 最大正方形 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/maximal-square/)

正方形的面积与边长直接相关

用`dp[i][j]`表示以`matrix[i][j]`为右下角的最大正方形边长

如果`matrix[i][j]=0`，那么这个正方形就不存在，边长为0

否则，最大正方形的边长与上一层的`dp[i-1][j-1]`，`dp[i-1][j]`，`dp[i][j-1]`相关，类似于木桶效应，需要取这三者的最小值，加1就得到`dp[i][j]`

状态转移方程为：
$$
dp[i][j] = \begin{cases}
0 &\text{matrix[i][j]=0}\\
min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1]) &\text{matrix[i][j]=1}\\
\end{cases}
$$
题解为`dp[i][j]`中的最大值的平方（题解是面积）

### 279.完全平方数

[279. 完全平方数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/perfect-squares/)

n为正整数，因此组成和为n的完全平方数的范围是1~n^(1/2)

首先初始化长度为 n+1 的数组 dp，每个位置都为 0
如果 n 为 0，则结果为 0
对数组进行遍历，下标为 i，每次都将当前数字先更新为最大的结果，即 dp[i]=i，比如 i=4，最坏结果为 4=1+1+1+1 即为 4 个数字

dp[i]的最坏情况为i，即每次都加1，dp[i]=dp[i-1]+1，因此将dp[i]初始化为i

状态转移方程为：`dp[i] = min(dp[i], dp[i - j * j] + 1)`

i 表示当前数字，j * j表示平方数，且 j 始终不超过 i

```java
for(int i=1;i<=n;i++)
{
    for(int j=1;j*j<=i;j++)
    {
        dp[i]=Math.min(dp[i],dp[i-j*j]+1);
    }
}
```

时间复杂度：O(n*sqrt(n))O(n∗sqrt(n))，sqrt 为平方根

### 300.最长递增子序列

[300. 最长递增子序列 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

定义dp[i]为以nums[i]结尾的最长子序列的长度

要从之前的 dp[j] ( j < i )中递推出 dp[i] 的值需要满足条件:

nums[i]>nums[j]，那么dp[i]=dp[j]+1，且dp[i]的值为满足这些条件的最大值

状态转移方程为
$$
dp[i]=max(dp[j]+1) （0=<j<i且nums[i]>nums[j]）
$$
所有的dp[i]初始化为1，表示子序列中只有一个元素nums[i]

最长递增子序列的长度从dp[i]中选取最大的

```java
		for(int i=0;i<n;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(nums[i]>nums[j])
                {
                    dp[i]=Math.max(dp[i],dp[j]+1);
                }
            }
            re=Math.max(re,dp[i]);
        }
```

### 309.最佳买卖股票时机含冷冻期

[309. 最佳买卖股票时机含冷冻期 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

每一天有三种可能的操作：买入，卖出，冷冻期

sell[i]表示截至第i天，（不持有股票）最后一个操作是卖时的最大收益；
buy[i]表示截至第i天，（持有股票）最后一个操作是买时的最大收益；
freeze[i]表示截至第i天，最后一个操作是冷冻期时的最大收益；

递推公式：
sell[i] = max(buy[i-1]+prices[i], sell[i-1]) (第一项表示第i天卖出，第二项表示第i天冷冻)
buy[i] = max(freeze[i-1]-prices[i], buy[i-1]) （第一项表示第i天买进，第二项表示第i天冷冻）
freeze[i] = max(sell[i-1], buy[i-1], freeze[i-1])

初始条件为：

sell[0]=0

buy[0]=-prices[0] 第一天就买入股票

freeze[0]=0

### 322.零钱兑换

[322. 零钱兑换 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/coin-change/)

采用自下而上的方法

dp[i] 表示凑成总金额为i所需的最少硬币数量

dp[i] 初始化为amount+1，表示不能凑成

但是dp[0]=0，表示边界情况，总金额为0时

第 j 个硬币面额为coin，则当i<coin时，其最少硬币数量与j-1种硬币时相同

当i>=coin时，其最少硬币数量在包含当前硬币coin和不包含coin中取最小

状态转移方程为：
$$
dp[i] = \begin{cases}
dp[i] &\text{i<coin}\\
min(dp[i-coin]+1,dp[i]) &\text{i>=coin}\\
\end{cases}
$$
若dp[amount] >amount，则说明无法凑成amount，题解为-1

否则说明可以凑成，且最少数量为dp[amount]

```java
 		for(int i=1;i<=amount;i++)
		{
            for(int j=0;j<n;j++)
            {
                int coin=coins[j];
                if(i>=coin)
                {
                    dp[i]=Math.min(dp[i-coin]+1,dp[i]);
                }
            }
		}
```

### 413.等差数列划分

[413. 等差数列划分 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/arithmetic-slices/)

利用等差数列的性质，对于一个等差数列来说，如果末尾新添加的元素与原来的末尾元素之间的差值和数列中的“差”相同，那么添加新的元素后，数列仍然是等差数列

因此使用dp[i]表示以nums[i]结尾的等差数列的数量

dp[0]和dp[1]为0，因为等差数列至少有三个元素

状态转移方程为：
$$
dp[i]=dp[i-1]+1 (i>2且nums[i]-nums[i-1]=nums[i-1]-nums[i-2])
$$
等差数列数量是将所有的dp[i]相加

具体代码：

```java
		int n=nums.length;
        if(n<3)
        {
            return 0;
        }
        int re=0;
        int[] dp=new int[n];
        for(int i=2;i<n;i++){
            if(nums[i-1]-nums[i-2]==nums[i]-nums[i-1])
            {
                dp[i]=dp[i-1]+1;
            }
            re+=dp[i];
        }
        return re;
```



### 416.分割等和子集

[416. 分割等和子集 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

本题与1049题采用同样的思路

考虑将数组分为两个子集的元素和是否相等

如果数组元素总和sum为奇数，那么绝对不可能存在，直接返回false

如果为偶数，考虑两个子集中的一个子集是否能达到m=sum/2，为方便dp，设该子集为较小的那一个，那么动态规划的目的就是让该子集的和取最大不断接近m

dp[i] [j] 表示数组前 i 个元素中，不大于 j 的最大值，该问题转化为0-1背包问题，num表示数组nums中的第 i 个元素

如果j<num，只能取前i-1个元素

否则，取前i-1个元素和加上当前元素的情况

状态转移方程为：
$$
dp[i][j] = \begin{cases}
dp[i-1][j] &\text{j<num}\\
max(dp[i-1][j],dp[i-1][j-num]+num) &\text{j>=num}\\
\end{cases}
$$
最后的结果需要判断dp[m]是否等于m

空间优化：

使用一维数组dp[m+1]，为了防止上一层循环的dp[0,.....,j-1]被覆盖，循环的时候 j 只能逆向遍历

```java
	   for(int i=0;i<n;i++)
       {
           int num=nums[i];
           for(int j=m;j>=num;j--)//必须逆序遍历
           {
               
                dp[j]=Math.max(dp[j],dp[j-num]+num);
               
           }
       }
```



### 486.预测赢家

[486. 预测赢家 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/predict-the-winner/)

对玩家来说，他需要尽可能地将他与另一名玩家分数的差值最大化

dp[i] [j] 表示当数组剩下的部分为下标 i 到下标 j 时，当前玩家与另一个玩家的分数之差的最大值，当前玩家不一定是先手

只有当i<=j时，数组剩下的部分才有意义，因此当i>j时，dp[i] [j]=0

当i=j时，dp[i] [i]表示只剩下第 i 个分数，那么玩家只能选择这一个，因此将dp[i] [i]初始化为nums[i]，其余的为0

对于当前的玩家来说，如果选择最左边的数即nums[i]，差值为nums[i]减去下一个玩家的dp[i+1] [j]，如果选择最右边的数即nums[j]，差值为nums[j]减去dp[i] [j-1]，这是根据留给下一个玩家的数组剩余部分而定的，而当前玩家需要选择这两者中最大的

状态转移方程为：
$$
dp[i][j]=max(nums[i]-dp[i+1][j],nums[j]-dp[i][j-1]) 当i<j
$$
我们看看状态转移的方向，它指导我们填表时采取什么计算方向，才不会出现：求当前的状态时，它所依赖的状态还没求出来。

dp[i] [j]依赖于dp[i+1] [j]和dp[i] [j-1]，因此 i 的值是从大到小，而 j 的值是从小到大，并且 j 要大于 i



遍历的顺序如下：

```java
for(int i=n-2;i>=0;i--)
{
    for(int j=i+1;j<n;j++)
    {
        dp[i][j]=Math.max(nums[i]-dp[i+1][j],nums[j]-dp[i][j-1]);
    }
}
```

最后的结果差值为dp[0] [n-1]

因为dp[i] [j]依赖于dp[i+1] [j]和dp[i] [j-1]，可以进行空间优化，遍历顺序照常

优化后的版本如下

```java
for(int i=n-2;i>=0;i--)
{
    for(int j=i+1;j<n;j++)
    {
        dp[j]=Math.max(nums[i]-dp[j],nums[j]-dp[j-1]);
    }
}
```



### 494.目标和

[494. 目标和 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/target-sum/)

数组求和为sum

假设数组中减的整数之和为neg

则有 target=sum-2*neg

从而 neg=(sum-target)/2，且sum-target必须为非负偶数（因为neg为非负整数）

将题目转化为nums数组中取任意个整数和为neg的方案数量

dp[i] [j]表示数组nums中前 i 个整数组成的结果为 j 的方案数量，dp[n] [neg]即为答案

边界情况为：

$$
dp[0][0]=1，dp[0][j]=0 (j>0)
$$

$$
dp[0][j] = \begin{cases}
1 &\text{j=0}\\
0 &\text{j>0}\\
\end{cases}
$$
对于nums中第i个整数num，如果 j 小于num则dp[i] [j]=dp[i-1] [j]，否则dp[i] [j]=dp[i-1] [j]+dp[i-1] [j-num]。

j<num时，num不能作为j的一部分加入，j>=num时，num既可以不加入，也可以加入

状态转移方程为：
$$
dp[i][j] = \begin{cases}
dp[i-1][j] &\text{j<num}\\
dp[i-1][j]+dp[i-1][j-num] &\text{j>=num}\\
\end{cases}
$$
题解为dp[n] [neg]

### 516.最长回文子序列

[516. 最长回文子序列 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

对于一个回文序列来说，如果在它的头尾各添加一个相同的字符，形成的新的序列仍然是回文序列

使用`dp[i][j]`表示字符串从第 i 到第 j 个字符之间最长的回文子序列长度。

如果第i个字符与第j个字符相同，则相当于第i+1个字符到第j-1个字符之间最长回文子序列首尾都增加一个相同的字符，`dp[i][j]`就是`dp[i+1][j-1]`+2

否则`dp[i][j]`取`dp[i+1][j]`和`dp[i][j-1]`的最大值，去掉头字符或去掉尾字符的最大长度

`dp[i][i]`初始化为1，表示字符串长度为1时，该字符串本身就是回文的，最长回文子序列长度为1

状态转移方程为：
$$
dp[i][j] = \begin{cases}
dp[i+1][j-1]+2 &\text{s(i)=s(j)}\\
max(dp[i+1][j],dp[i][j-1]) &\text{s(i)!=s(j)}\\
\end{cases}
$$
题解为dp[0] [n-1]

遍历的顺序需要注意，由于`dp[i][j]`需要通过`dp[i+1][j-1]`,`dp[i+1][j]`和`dp[i][j-1]`

得到，因此 i 需要从大到小遍历(n-1到0)，而 j 需要从小到大遍历（i+1到n-1）

具体代码如下:

```java
		int n=s.length();
        int[][] dp=new int[n][n];
        for(int i=0;i<n;i++)
        {
            dp[i][i]=1;
        }
        for(int i=n-1;i>=0;i--)
        {
            for(int j=i+1;j<n;j++)
            {
                if(s.charAt(i)==s.charAt(j))
                {
                    dp[i][j]=dp[i+1][j-1]+2;
                }
                else
                {
                    dp[i][j]=Math.max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
```



### 518.零钱兑换II

[518. 零钱兑换 II - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/coin-change-2/)

每种面额的硬币无限个

转化为完全背包问题

dp[i] [j] 为前 i 种硬币凑出总金额为 j 的方案数量

第i种硬币的面额为coin

当 j <coin时，j 不能选择第 i 种硬币，因此方案数量与前i-1种总金额为j的相同

当 j >=coin时，j 可以选择第i种硬币，而且由于每种硬币有无限个，因此方案数量包括前 i-1 种总金额为 j，和前 i 种总金额为 j-coin 的方案数量

状态转移方程为：
$$
dp[i][j] = \begin{cases}
dp[i-1][j] &\text{j<coin}\\
dp[i-1][j]+dp[i][j-coin] &\text{j>=coin}\\
\end{cases}
$$
题解为dp[n] [amount]

空间优化方案

和01背包问题类似，也可进行空间优化，优化后不同点在于这里的 j 只能**正向枚举**而01背包只能逆向枚举，因为这里的第二项是`dp[i]`而01背包是`dp[i-1]`，即这里就是需要覆盖而01背包需要避免覆盖。所以伪代码如下：

```text
// 完全背包问题思路一伪代码(空间优化版)
for i = 0,...,n-1
    for j = coins[i],...,amount // 必须正向枚举!!!
        dp[j] = dp[j]+dp[j−coins[i]]
```

由上述伪代码看出，01背包和完全背包问题此解法的空间优化版解法唯一不同就是前者的 j 只能逆向枚举而后者的 j 只能正向枚举，这是由二者的状态转移方程决定的。此解法时间复杂度为O(n*amount), 空间复杂度为O(amount)。

### 524.通过删除字母匹配到字典里最长单词

[524. 通过删除字母匹配到字典里最长单词 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

使用双指针匹配花费大量时间在s中寻找对应的字符，而且重复了多次

因此想办法将这个过程简化，如果将字符在s中的位置存储起来就可以省去大量时间

因此使用`dp[i][j]`表示字符串从下标 i 开始字符（'a'+j）出现的第一个位置，如果没有该字符则`dp[i][j]`=m，即字符串s的长度，表示下标 i **及之后**没有该字符

`dp[m+1][26]`，初始化dp[m]的值都为m，从后往前遍历数组dp

如果字符串s的第i个字符恰好为'a'+j，则`dp[i][j]=i`，否则`dp[i][j]=dp[i+1][j]`，即在下标 i **之后**字符'a'+j第一次出现的位置，所以需要从后往前遍历数组

状态转移方程为：
$$
dp[i][j] = \begin{cases}
i &\text{s[i]='a'+j}\\
dp[i+1][j] &\text{s[i]!='a'+j}\\
\end{cases}
$$
对字符串预处理的过程如下：

```java
		int[][] dp=new int[m+1][26];
        Arrays.fill(dp[m],m);
        for(int i=m-1;i>-1;i--)
        {
            for(int j=0;j<26;j++)
            {
                if(s.charAt(i)==(char)j+'a')
                {
                    dp[i][j]=i;
                }
                else
                {
                    dp[i][j]=dp[i+1][j];
                }
            }
        }
```

得到处理好的dp数组之后，遍历字典，匹配字符串cur

从前往后遍历字符串cur，寻找cur当前字符在s中下标index及之后的位置temp，如果是m，则不匹配。而且index需要不断更新，具体做法是更新为temp+1，表示下一次需要从s中temp的位置之后开始寻找字符

匹配完成后将re更新为最长且字典序最小的字符串cur

```java
		for(String cur:dictionary){
            int curlen=cur.length();
            boolean flag=true;
            int index=0;
            for(int i=0;i<curlen;i++)
            {
                int temp=dp[index][cur.charAt(i)-'a'];
                if(temp==m)
                {
                    flag=false;
                    break;
                }
                index=temp+1;
            }
            if(flag)
            {
                int relen=re.length();
                if(curlen>relen||(curlen==relen&&cur.compareTo(re)<0))
                {
                    re=cur;
                }
            }
        }
```



### 647.回文子串

[647. 回文子串 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/palindromic-substrings/)

与5题相同的解法，本题求的是回文子串的数量，所以在`dp[i][j]=true`时，计数器加一即可

### 787.K站中转内最便宜的航班

[787. K 站中转内最便宜的航班 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/)

最多经过k站中转，等同于最多乘坐k+1次航班（即走k+1步）

`dp[i][j]`表示走了 i 步到达城市j的最小花费

对于航班flight来说，到达城市to的花费可以到达通过城市from加上该flight的price得到，因此可以逐步遍历所有航班得到最小花费

初始化dp：

最开始`dp[i][j]`=MAX(表示花费无穷大，不可达)

但是`dp[0][src]`=0，表示src为出发点

状态转移方程为：
$$
dp[i][to] = \begin{cases}

min(dp[i][to],dp[i-1][from]+price) &\text{flight(from,to,price)}\\
\end{cases}
$$
题解为dp[i] [dst]中的最小值

空间优化，`dp[i][j]`只与`dp[i-1][j]`相关，因此可以优化为一维数组储存状态，但是需要两个数组储存，确保当前状态不是由当前步数下的其他状态得到，而是由上一步下的其他状态得到。

### 879.盈利计划

[879. 盈利计划 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/profitable-schemes/)

因为约束条件包括工作成员总数n和至少产生的利润minProfit，所以采用三维的动态规划`dp[group.length+1][n+1][minProfit+1]`

`dp[i][j][k]` 表示前 i 组员工在工作成员总数为 j 的情况下至少产生利润为 k 的方案数量，

初始化值为dp[0] [0] [0]=1，表示至少产生利润为0的方案有一种，就是不派遣任何工人

第 i 个组的人数为group，对应的利润为profit

当j<group时，不能派遣第 i 组，只能和前i-1个组的方案数量相同

当j>=group时，可以也可以不派遣第 i 组，方案数量包括前i-1个组的方案数量和派遣这一组的方案数量，即前i-1个组的基础上总工作人数为j-group，且至少获得的利润由k的大小决定，如果k>profit，那就是k-profit，否则就是0，

状态转移方程为：
$$
dp[i][j][k] = \begin{cases}
dp[i-1][j][k] &\text{j<group}\\
dp[i-1][j][k]+dp[i-1][j-group][max(k-profit,0)] &\text{j>=group}\\
\end{cases}
$$
根据题目要求，所有的产生利润至少为minProfit的方案数量需要将 j=0~n 的dp[group.length] [j] [minProfit] 求和

### 887.石子游戏

[877. 石子游戏 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/stone-game/)

与 486.预测赢家 类似

但是石子游戏中限制石子堆数量为偶数，且石子总数量为奇数，那么先发玩家必定赢

直接return true

### 918.环形子数组的最大和

[918. 环形子数组的最大和 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/maximum-sum-circular-subarray/)

本题相比于53题多了环形的条件

因此将最终的结果分为不在环形中和在环形中两种情况

不在环形中：与53题同，就是数组nums的最大连续子数组之和max：

`dp[i]=Math.max(dp[i-1],0)+nums[i]`

`max=Math.max(dp[0],dp[1],...,dp[n-1])`

在环形中：必然包含nums[n-1]和nums[0]，如果这是环形数组的最大连续子数组之和，那么在nums[1]~nums[n-2]中必定有负数，否则不需要按照从数组尾到数组头的顺序组建子数组

因此将nums[2]~nums[n-2]段的最小连续子数组之和min求出，然后用nums数组之和sum减去min，就是在环形中情况的最大值

`dp[i]=MAth.min(dp[i-1],0)+nums[i]`

`min=Math.min(dp[0],dp[1],...,dp[n-1])`

最后的结果需要在两种情况中取较大的，即`Math.max(sum-min,max)`

### 1014.最佳观光组合

[1014. 最佳观光组合 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/best-sightseeing-pair/)

观光组合（j,i）中第一个景点的分数为`values[j]+j`

第二个景点的分数为`values[i]-i`

考虑dp[i]表示观光组合中第二个观光景点为 i 时的最大评分

现在需要维持观光组合中第一个景点分数最大`first=values[j]+j (j=0,...,i-1)`

因此在遍历观光景点数组时，需要更新max的值，只需要取最大的dp[i]就可以得到观光景点组合的最高分

状态转移方程为：



$$
dp[i]=first+values[i]-i
$$

$$
first=max(first,values[i]+i)
$$

可以看出，dp[i]只与上一次的first有关，空间优化后为：

```java
for(int i=0;i<n;i++)
{
    re=Math.max(re,first+values[i]-i);
    first=Math.max(first,values[i]+i);//维持观光组合中的第一个观光景点最大
}
```



### 1049.最后一块石头的重量II

[1049. 最后一块石头的重量 II - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/last-stone-weight-ii/)

将stones中的石头分为两堆，两堆的重量差值最小

即每一堆的重量接近于石头总重量的一半sum/2

将原问题转化为0-1背包问题

`dp[i][j]` 表示从 i 个石头取任意个石头中最接近 j 的总重量，最接近指<=j，方便动态规划，在保证重量不超过 j 的前提下，此后的最接近就是最大

stone表示第i个石头的重量stones[i-1]，当j<stone时，不能将stone计入重量，取 i-1 个石头最接近 j 的重量；当j>=stone时，可以将stone计入重量，且为保持最接近 j ，应当是取`max(dp[i-1][j],dp[i-1][j-stone]+stone)`

状态转移方程为：
$$
dp[i][j] = \begin{cases}
dp[i-1][j] &\text{j<stone}\\
max(dp[i-1][j],dp[i-1][j-stone]+stone) &\text{j>=stone}\\
\end{cases}
$$
`dp[n][sum/2]`即为最接近总重量一半的重量

题解为`sum-2*dp[n][sum/2]`

空间优化方案：

由上述状态转移方程可知，`dp[i][j]`的值只与`dp[i-1][0,...,j-1]`有关，所以我们可以采用动态规划常用的方法（滚动数组）对空间进行优化（即去掉dp的第一维）。需要注意的是，为了防止上一层循环的`dp[0,...,j-1]`被覆盖，循环的时候 j 只能**逆向枚举**（空间优化前没有这个限制），伪代码为：

```text
// 01背包问题伪代码(空间优化版)
dp[0,...,sum/2] = 0
for i = 1,...,n
    for j = sum/2,...,stones[i] // 必须逆向枚举!!!
        dp[j] = max(dp[j], dp[j−stones[i]]+stones[i])
```

取m=sum/2，时间复杂度为O(nm), 空间复杂度为O(m)。由于m的值是m的位数的幂，所以这个时间复杂度是伪多项式时间。

动态规划的核心思想**避免重复计算**在01背包问题中体现得淋漓尽致。第i个石头装入或者不装入而获得的最大重量完全可以由前面i-1件物品的最大重量决定，暴力枚举忽略了这个事实。

### 1277.统计全为1的正方形子矩阵

[1277. 统计全为 1 的正方形子矩阵 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/)

本题与221题相同

需要正方形子矩阵的数量，考虑到以maxtrix[i] [j]为右下角的正方形数量就等于其最大正方形边长，而且这样各个子正方形矩阵是不会重复的，也不会遗漏，遍历矩阵中每一个元素将其相加就可以得到所有的正方形子矩阵的数量

## 递归回溯法

能用回溯法解决的求方案数量的问题可以用动态规划求解

求方案具体内容用回溯法

![转化过程](https://pic.leetcode-cn.com/2cdf411d73e7f4990c63c9ff69847c146311689ebc286d3eae715fa5c53483cf-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-03-08%2010.23.03.png)

### 剑指Offer 38.字符串的排列

题目要求字符串数组中不能有重复元素所以需要去重操作

采用最基础的回溯法，如果构成的字符串cur长度与原字符串s相同，则说明cur是一个可行解，加入结果集set中，否则需要继续构造

为了避免同一个字符重复加入cur，使用visited数组表示是否访问过对应下标的字符，如果访问过，则跳过，否则将其加入cur进行递归回溯

```java
		if(cur.length()==s.length())
        {
            set.add(new String(cur));
            return;
        }
        for(int i=0;i<s.length();i++)
        {
            if(visited[i])
            {
                continue;
            }
            else
            {
                cur.append(s.charAt(i));
                visited[i]=true;
                backtrack(s,cur,visited);
                visited[i]=false;
                cur.deleteCharAt(cur.length()-1);
            }
        }
```

但是该递归函数并没有满足「全排列不重复」的要求，在重复的字符较多的情况下，该递归函数会生成大量重复的排列。对于任意一个空位，如果存在重复的字符，该递归函数会将它们重复填上去并继续尝试导致最后答案的重复。

解决该问题的一种较为直观的思路是，我们首先生成所有的排列，然后进行去重，即使用Set类型的结果集自动去重。而另一种思路是我们通过修改递归函数，使得该递归函数只会生成不重复的序列。

具体地，我们只要在递归函数中设定一个规则，保证在填每一个空位的时候重复字符只会被填入一次。具体地，我们首先对原字符串**排序**，保证相同的字符都相邻，在递归函数中，我们限制每次填入的字符一定是这个字符所在重复字符集合中**「从左往右第一个未被填入的字符」**，即如下的判断条件：

```java
if(visited[i]||(i>0&&arr[i]==arr[i-1]&&!visited[i-1]))
{
    continue;
}
```

`i>0`是为了保证不超出边界，`arr[i]==arr[i-1]`表示当前字符与前一个字符相同，而`!visited[i-1]`表示前一个字符未被访问过，这种情况表明在前一个字符`arr[i-1]`未被填入该空位的情况下填入当前相同的字符`arr[i]`，那必然与将前一个字符`arr[i-1]`填入该空位的情况重复，这种重复的情况必然已经被计算过，所以需要跳过这种情况

加入此限制条件后，就可以保证是从左到右依次将重复的字符填入空位中

### 79.单词搜索

[79. 单词搜索 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/word-search/)

遍历网格数组中的每一个位置，从该位置开始进行上下左右的移动，进而判断能否组成word。只要有一个位置出发能组成word，就返回true，否则继续进行下一个网格的dfs。

同时维护一个与网格相同大小的visited数组标记当前位置是否已经被访问过。

深度优先遍历函数中

如果当前位置的字符与word[index]位置的字符不同则直接返回false

如果当前位置字符与word最后一个字符相同，说明已经完成了遍历，返回true

如果当前位置字符与word[index]位置的字符相同，但又不是最后一个，则继续进行上下左右的移动

每次移动，首先判断移动后的新位置是否越界，以及该位置是否已经 被访问过

如果越界或者已经被访问，则跳过该位置，不需要进行dfs

否则将该位置的标记为已经被访问，进行dfs，判断后续字符是否与word匹配。判断完之后（如果匹配直接返回true，否则才需要回溯），将该位置标记为未访问，这就达到了回溯的效果

### 212.单词搜索II

[212. 单词搜索 II - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/word-search-ii/)

最直接就是遍历单词字典的每一个单词，调用79题中的判定函数，逐个判断是否能在网格中搜索到对应的单词，如果能搜索到就加入结果列表，但是这样做效率很低，进行了大量重复的遍历网格操作

为了减少对网格的遍历，使用set存储字典里的单词

```java
		for(String word:words)
        {
            set.add(word);
        }
```

在dfs搜索网格的过程中，维护一个StringBuilder类型的字符串cur，cur通过I在网格中移动并扩充对应的字符得到（首先需要遍历网格，从网格的每一个位置出发开始移动搜索）

```java
		for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                visited[i][j]=true;
                StringBuilder cur=new StringBuilder();
                cur.append(board[i][j]);
                dfs(cur,i,j);
                visited[i][j]=false;
            }
        }
```

dfs过程：

根据题目所给条件，字典中的单词长度不超过10，因此在遍历过程中，如果cur长度超过10，就直接返回，进行剪枝

如果发现cur被set包含，则将其加入结果列表，并在set中移除cur字符串，这是为了在结果中去重

```java
		if(cur.length()>10)
        {
            return ;
        }
        String temp=cur.toString();
        if(set.contains(temp))
        {
            re.add(temp);
            set.remove(temp);
        }
```

后续的移动过程与79题类似，上下左右移动，进行越界和是否被访问判断，回溯也类似，只是因为加入了cur字符串，所以需要对字符串进行扩充和“去尾”以达到回溯的效果

```java
		for(int i=0;i<4;i++)
        {
            int newx=x+move[i][0];
            int newy=y+move[i][1];
            if(newx<0||newx>=m||newy<0||newy>=n||visited[newx][newy])
            {
                continue;
            }
            visited[newx][newy]=true;
            cur.append(board[newx][newy]);
            dfs(cur,newx,newy);
            cur.deleteCharAt(cur.length()-1);
            visited[newx][newy]=false;
        }
```



### 322.零钱兑换

[322. 零钱兑换 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/coin-change/)

记忆化搜索

可以看出在进行递归的时候，有很多重复的节点要进行操作，这样会浪费很多的时间。
使用数组 count[ ]来保存节点的值
count[n] 表示钱币 n 可以被换取的最少的硬币数，不能换取就为 -1
递归函数的目的是为了找到amount 数量的零钱可以兑换的最少硬币数量，返回其值 int

在进行递归的时候，count[n]被复制了，就不用继续递归了，可以直接的调用

### 486.预测赢家

[486. 预测赢家 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/predict-the-winner/)

对于玩家1与玩家2获得分数的差值，玩家1需要最大化这个差值，而玩家2需要最小化这个差值

每个玩家在选择数组中最左还是最右的数时，需要考虑的是如何将差值最大或最小化，因此采用递归的方式，round表示当前玩家，1表示玩家1，-1表示玩家2，这样也方便计算差值，所得分数乘以round求和即为差值

```java
	private int dfs(int[] nums,int left,int right,int round)
    {
        if(left==right)
        {
            return nums[left]*round;
        }
        int leftscore=nums[left]*round+dfs(nums,left+1,right,-round);
        int rightscore=nums[right]*round+dfs(nums,left,right-1,-round);
        if(round>0)//玩家1
        {
            return Math.max(leftscore,rightscore);
        }
        else//玩家2
        {
            return Math.min(leftscore,rightscore);
        }
    }
```

left等于right时，表示这时数组中只剩下一个数

leftscore表示当前玩家选择最左边的数所得到的差值

rightscore表示当前玩家选择最右边的数得到的差值



更直接的理解是，当前玩家需要将自己的分数与对手玩家的分数差值最大化，

```java
	private int dfs(int[] nums,int left,int right)
    {
        if(left==right)
        {
            return nums[left];
        }
        int leftscore=nums[left]-dfs(nums,left+1,right);
        int rightscore=nums[right]-dfs(nums,left,right-1);
        return Math.max(leftscore,rightscore);
        
    }
```

采用记忆化搜索将其改进，将计算过的结果存储在数组memo[n] [n] 中，当前的数组剩余部分为下标left到right，如果memo[left] [right]已经赋值过则可以直接返回其值，这样就避免了重复计算

```java
 	private int dfs(int[] nums,int left,int right)
    {
        if(left==right)
        {
            return nums[left];
        }
        if(memo[left][right]!=Integer.MIN_VALUE)//已经计算过memo[left][right]的值
        {
            return memo[left][right];
        }
        int leftscore=nums[left]-dfs(nums,left+1,right);
        int rightscore=nums[right]-dfs(nums,left,right-1);
        memo[left][right]=Math.max(leftscore,rightscore);
        return memo[left][right];
    }
```



### 494.目标和

[494. 目标和 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/target-sum/)

2^n

对于nums数组中的每个整数，可以加或者减

因此共有2^n种情况

这些情况中遍历完数组求和等于target的才是符合题设要求

### 526.优美的排列

[526. 优美的排列 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/beautiful-arrangement/)

我们可以使用回溯法解决本题，从左向右依次向目标排列中放入数即可。

具体地，我们定义函数 backtrack(n,index,visited)，表示尝试向位置 index 放入数。其中 n 表示排列的长度。在当前函数中，我们首先找到一个符合条件的未被使用过的数，然后递归地执行backtrack(n,index+1,visited)，当该函数执行完毕，回溯到当前层，我们再尝试下一个符合条件的未被使用过的数即可。

回溯过程中，我们可以用visited数组标记哪些数被使用过，每次我们选中一个数 i，我们就将 visited[i] 标记为true，回溯完成后，我们再将其置为false。

## DFS

深度优先遍历

### 200.岛屿数量

[200. 岛屿数量 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/number-of-islands/)

将二维网格看成一个无向图，竖直或水平相邻的 1 之间有边相连。

为了求出岛屿的数量，我们可以扫描整个二维网格。如果一个位置为 1，则以其为起始节点开始进行深度优先搜索。在深度优先搜索的过程中，每个搜索到的 1都会被重新标记为 0，这样就可以唯一确定每一个岛屿。

最终岛屿的数量就是我们进行深度优先搜索的次数。

### 538.把二叉搜索树转换为累加树

[538. 把二叉搜索树转换为累加树 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

二叉搜索树是一棵空树，或者是具有下列性质的二叉树：

若它的左子树不空，则左子树上所有节点的值均小于它的根节点的值；

若它的右子树不空，则右子树上所有节点的值均大于它的根节点的值；

它的左、右子树也分别为二叉搜索树。

由这样的性质我们可以发现，二叉搜索树的中序遍历是一个单调递增的有序序列。如果我们反序地中序遍历该二叉搜索树，即可得到一个单调递减的有序序列。

所以可以反序中序遍历二叉搜索树，向右再根最后左

累加这个顺序遍历的节点值，就可以转换为累加树

```java
	int sum=0;
    public TreeNode convertBST(TreeNode root) {
        if(root!=null)
        {
            convertBST(root.right);
            sum+=root.val;
            root.val=sum;
            convertBST(root.left);
        }
        return root;
    }
```

## 哈希表

### 128.最长连续序列

[128. 最长连续序列 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

使用集合set去重 

对于一个连续序列，按照从小到大的顺序排列，开头的元素为x，则x-1一定不在集合set中，因此遍历集合寻找开头元素x，然后在集合set中继续寻找序列后面的元素，即x+1，x+2，x+3，......，从而得到序列长度，结果取序列长度的最长值

### 930.和相同的二元子数组

[930. 和相同的二元子数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/binary-subarrays-with-sum/)

用哈希表存放数组nums的前缀和，及相应的数量

前缀和sum初始化为0

在遍历数组的时候将前缀和放入哈希表中，再计算当前的前缀和，不断累加哈希表中sum-goal的数量就可以得到最终结果

关键在于顺序，首先将前缀和加入哈希表是为了避免重复的同时，不遗漏从数组下标为0的元素开始的情况，sum-goal说明当前的前缀和需要减去的之前一段前缀和的大小

```java
		for(int i=0;i<n;i++)
        {
            map.put(sum,map.getOrDefault(sum,0)+1);
            sum+=nums[i];
            re+=map.getOrDefault(sum-goal,0);
            
        }
```



### 981.基于时间的键值存储

[981. 基于时间的键值存储 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/time-based-key-value-store/)

Map<String,TreeMap<Integer,String>> map存储基于时间的键值对

使用TreeMap是为了自动排序

Map的方法computeIfAbsent()[(1条消息) map中的computeIfAbsent方法_u010659877的专栏-CSDN博客](https://blog.csdn.net/u010659877/article/details/77895080)

如果map中不存在key，则按参数中的function构造键值对put进去

TreeMap的方法floorEntry(key)，返回哈希表中键值不小于key的键值对

### 1711.大餐计数

[1711. 大餐计数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/count-good-meals/)

建立哈希表map存储大餐的美味程度和数量

遍历数组时首先判断哈希表中是否存在与当前元素相加等于2的幂的元素，如果有，就将结果加上该元素对应的数量，这样的元素可能有多个，所以选择遍历所有的2的幂，而不是遍历哈希表的所有元素（由于是int类型，2的幂有32个，而哈希表中的元素数量可能很多，而且还需要逐个判断是否相加为2的幂，这样是为了防止超时）

最后再将该元素加入哈希表

重点：这样当前元素的下标大于哈希表中所有元素的下标，从而避免了重复的情况

```java
	public int countPairs(int[] deliciousness) {
        int m=1000000007;
        int re=0;
        HashMap<Integer,Integer> map=new HashMap<>();
        for(int deli:deliciousness)
        {
            for(int i=0;i<32;i++)
            {
                int temp=(1<<i)-deli;
                re=(re+map.getOrDefault(temp,0))%m;
            }
            map.put(deli,map.getOrDefault(deli,0)+1);
        }
        return re;
    }
```

### 863.二叉树中所有距离为K的结点

[863. 二叉树中所有距离为 K 的结点 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/)

每个结点最多有三个结点与其相连，左子结点，右子结点和父结点，对于二叉树原本的结构来说，只有父结点是无法直接访问的，可以使用哈希表parents存储每个结点和其父结点的对应关系，遍历二叉树，得到所有结点（根结点可除外）的父结点

```java
	HashMap<TreeNode,TreeNode> parents=new HashMap<>();

	private void findparent(TreeNode node,TreeNode parent)
    {
        if(node==null)
        {
            return;
        }
        parents.put(node,parent);
        findparent(node.left,node);
        findparent(node.right,node);
    }
```

之后从目标结点target开始进行dfs，遍历所有距离target为5 的结点并保存

为了减少遍历时间，采用记忆化搜索，使用HashSet保存已经被访问过的结点

```java
HashSet<Integer> visited=new HashSet<>();
```

每次dfs时，都要判断当前结点是否在visited中，如果在，说明结点已经被访问，则不用dfs，直接返回，否则继续后面的判断操作。每进行一次dfs，步数减一，当步数为0时，说明当前结点就是符合条件的，将其加入结果列表中，dfs具体代码如下：

```java
	private void dfs(TreeNode node,int k)
    {
        if(node==null)
        {
            return;
        }
        if(visited.contains(node.val))
        {
            return;
        }
        visited.add(node.val);
        if(k==0)
        {
            re.add(node.val);
            return;
        }
        dfs(node.left,k-1);
        dfs(node.right,k-1);
        dfs(parents.get(node),k-1);
    }
```



## 投票算法

### 面试题 17.10.主要元素

[面试题 17.10. 主要元素 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-majority-element-lcci/)

要求数组nums中占比超过一半的元素

将候选人初始化为数组中下标为0的元素：candidate=nums[0]

对应的计票为1：count=1

从下标为1的元素开始遍历数组，每一个元素都有可能成为候选人，且每个元素仅投票给与自己相同的候选人

如果当前计票为0，则将候选人改为当前元素

如果当前元素等于（支持）候选人，则计票加一

如果不等于（支持），则计票减一

遍历完数组后得到的候选人，需要再次遍历整个数组，统计该候选人的总票数count

再判断count*2是否大于数组长度，这是为了防止出现最后选中的候选人是数组中最后一个元素但仍不符合占比超过一半

```java
		for(int i=1;i<n;i++)
        {
            if(count==0)
            {
                candidate=nums[i];
            }
            if(nums[i]==candidate)
            {
                count++;
            }
            else
            {
                count--;
            }
        }
        count=0;
        for(int num:nums)
        {
            if(candidate==num)
            {
                count++;
            }
        }
        return count*2>n?candidate:-1;
```

  
