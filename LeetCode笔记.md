# LeetCode笔记

## 贪心

### [397. 整数替换](https://leetcode-cn.com/problems/integer-replacement/)

我们可以从二进制的角度进行分析：给定起始值 n，求解将其变为 (000...0001)_2的最小步数。

对于偶数（二进制最低位为 0）而言，我们只能进行一种操作，其作用是将当前值 x 其进行一个单位的右移；
对于奇数（二进制最低位为 1）而言，我们能够进行 +1 或 -1 操作，分析两种操作为 x 产生的影响：
对于 +1 操作而言：最低位必然为 1，此时如果次低位为 0 的话， +1 相当于将最低位和次低位交换；如果次低位为 1 的话，+1 操作将将「从最低位开始，连续一段的 1」进行消除（置零），并在连续一段的高一位添加一个 1；
对于 -1 操作而言：最低位必然为 1，其作用是将最低位的 1 进行消除。
因此，对于 x 为奇数所能执行的两种操作，+1 能够消除连续一段的 1，只要次低位为 1（存在连续段），应当优先使用 +1 操作，但需要注意边界 x = 3 时的情况（此时选择 -1 操作）。

```java
	public int integerReplacement(int n) {
        int count=0;
        long num=n;//防止边界值溢出
        while(num!=1){
            if((num&1)==1){//n为奇数
                if(num!=3&&(num&2)==2){
                    num++;
                }else{
                    num--;
                }
            }else{
                num=num>>1;
            }
            count++;
        }
        return count;
    }
```



### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

​		求最少的移除区间个数，等价于尽量多保留不重叠的区间。在选择要保留区间时，区间的结

尾十分重要：选择的区间结尾越小，余留给其它区间的空间就越大，就越能保留更多的区间。因

此，我们采取的贪心策略为，优先保留结尾小且不相交的区间。

​		具体实现方法为，先把区间按照结尾的大小进行增序排序，每次选择结尾最小且和前一个选

择的区间不重叠的区间。

​		在样例中，排序后的数组为 [[1,2], [1,3], [2,4]]。按照我们的贪心策略，首先初始化为区间

[1,2]；由于 [1,3] 与 [1,2] 相交，我们跳过该区间；由于 [2,4] 与 [1,2] 不相交，我们将其保留。因

此最终保留的区间为 [[1,2], [2,4]]。

```java
	public int eraseOverlapIntervals(int[][] intervals) {
        if(intervals.length == 0) return 0;
        Arrays.sort(intervals, new Comparator<int[]>() {
		    public int compare(int[] a, int[] b){
			    return a[1]-b[1];
		    }
	    });
        int n=intervals.length;
        int re=0;
        int prev=0;
        for(int i=1;i<n;i++){
            if(intervals[i][0]<intervals[prev][1])
            {
                re++;
            }
            else{
                prev=i;
            }
        }
        return re;
    }
```



### [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/)

基本思路：最多只能改变一个元素，那么遍历数组，遇到第一个不满足非递减的元素（nums[i]>nums[i+1]）改动元素，继续遍历，如果后续还有不满足非递减就返回false，直到遍历完整个数组返回true

关键在于改动元素的方法，改动哪个元素？把元素改为多少？

考虑最基本的情况：4，2，3

遍历到4时，4>2，按照非递减原则，将4改为2就可以满足，这里就是将nums[i]改为更小的nums[i+1]

但是这种改动方法在遇到如下情况时会出错：5，7，1，8

遍历到7时，7大于1，如果将7改为1，那么前面的5就大于1，不满足非递减，而如果将1改为7就完全满足非递减

因此这时改动的是将nums[i+1]改为nums[i]，这种情况是前面的5仍然大于1，即nums[i-1]>nums[i+1]

这样就得到了改动元素的方法，保证元素不小于它之前的元素

```java
		int n=nums.length;
        int i=0;
        while(i<n-1){
            if(nums[i]>nums[i+1]){
                if(i>0&&nums[i-1]>nums[i+1]){
                    nums[i+1]=nums[i];
                }
                else
                {
                    nums[i]=nums[i+1];
                }
                break;
            }
            i++;
        }
        while(i<n-1){
            if(nums[i]>nums[i+1]){
                return false;
            }
            i++;
        }
        return true;
```



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

### [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

旋转排序数组可以被拆分为2个排序数组nums1，nums2，并且nums1任一元素>=nums2任一元素；因此，考虑二分法寻找这两个数组的分界点nums[i]（即第二个数组nums2的首个元素）

设置left,right指针在数组两端，mid为每次二分的中点

当 nums[mid] > nums[right]时，mid 一定在第 1 个排序数组中，i 一定满足 mid < i <= right，因此执行 left = mid + 1；
当 nums[mid] < nums[right] 时，mid 一定在第 2 个排序数组中，i 一定满足 left < i <= mid，因此执行 right = mid；
当 nums[mid] == nums[right] 时，是此题对比 153题 的难点（原因是此题中数组的元素可重复，难以判断分界点 i 指针区间）；
例如 [1, 0, 1, 1, 1][1,0,1,1,1] 和 [1, 1, 1, 0, 1][1,1,1,0,1] ，在 left = 0, right = 4, mid = 2 时，无法判断 mid 在哪个排序数组中。
我们采用 right = right - 1 解决此问题，证明：
		此操作不会使数组越界：因为迭代条件保证了 right > left >= 0；
		此操作不会使最小值丢失：假设nums[right] 是最小值，有两种情况：
				若 nums[right] 是唯一最小值：那就不可能满足判断条件 nums[mid] == nums[right]，因为 mid < right（left != right 且 mid = (left + right) // 2 向下取整）；
				若 nums[right] 不是唯一最小值，由于 mid < right 而 nums[mid] == nums[right]，即还有最小值存在于 [left, right - 1][left,right−1] 区间，因此不会丢失最小值。



### 162.寻找峰值

[162. 寻找峰值 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/find-peak-element/)

找到左边和右边相邻元素都小于自身的元素

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



### [475. 供暖器](https://leetcode-cn.com/problems/heaters/)

每个房屋要么被左边的供暖器供暖，要么被右边的供暖，因此需要找到离房屋最近的供暖器，在所有房屋的这个距离中取最大的就是最小供暖半径x

排序+二分查找

首先对供暖器排序，使用二分查找找到房屋左边的最近的供暖器heater[index]，从而heater[index+1]就是房屋右边的最近的供暖器，在这两个距离中取较小的

```java
	private int binarySearch(int[] heaters,int house){//找到房屋左边的最近的供暖器
        int re=0;
        int left=0;
        int right=heaters.length-1;
        while(left<=right){
            int mid=(left+right)>>1;
            if(heaters[mid]>=house){
                right=mid-1;
            }
            else {
                left=mid+1;
                re=mid;
            }
        }
        return re;
    }
```



同时需要考虑边界情况：所有供暖器都在房屋右边，所有供暖器都在房屋左边

```java
		Arrays.sort(heaters);
        int x=0;
        for(int house:houses){
            int index=binarySearch(heaters,house);
            if(heaters[index]>house){//所有供暖器都在房屋右边
                x=Math.max(x,heaters[index]-house);
            }else{
                if(index==heaters.length-1){//所有供暖器都在房屋左边
                    x=Math.max(x,house-heaters[index]);
                }else{
                    int temp=Math.min(house-heaters[index],heaters[index+1]-house);//在左右两边找离房屋最近的供暖器
                    x=Math.max(x,temp);
                }
            }
        }
        return x;
```



### [540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

位运算：把数组中的每个元素异或，最后的结果就是唯一元素，时间复杂度为O(n)

要想达到O(logn)的时间复杂度就要使用二分查找

使用left和right指向数组两端，每次二分的中点为mid，

考虑如下性质：

如果nums[mid]不等于nums[mid+1]也不等于nums[mid-1]，则nums[mid]就是唯一元素，直接返回

否则就要考虑比nums[mid]小的元素个数less，如果less是偶数，则唯一元素比nums[mid]大，在其右边，如果less是奇数，则唯一元素比nums[mid]小，在其右边

通过这样一个性质就可以写出二分范围缩小的条件，另外还需要注意边界值0和n-1

```java
		while(left<=right){
            int mid=left+(right-left)/2;
            if(mid>0&&nums[mid]==nums[mid-1]){
                less=mid-1;
            }
            else if(mid<n-1&&nums[mid]==nums[mid+1]){
                less=mid;
            }else{
                return nums[mid];
            }
            if((less&1)==0){//比nums[mid]小的元素有偶数个，则唯一一个元素在右边
                left=mid+1;
            }else{
                right=mid-1;
            }
        }
```



### 704.二分查找

[704. 二分查找 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/binary-search/)

最基础的二分查找



## 位运算

二进制的性质

巧用异或

### [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

首先想到判断每一位是否为1，即将n与对应位数为1的数(1<<i)做与运算，结果不为0就表示n的该位上是1

```java
		int re=0;
        for(int i=0;i<32;i++)
        {
            if((n&(1<<i))!=0)//1左移i位就是2^i
            {
                re++;
            }
        }
        return re;
```

优化：

观察这个运算：n&(n - 1)，其预算结果恰为把 n 的二进制位中的最低位的 1 变为 0 之后的结果。

如：6&(6-1) = 4, 运算结果 4 即为把 6 的二进制位中的最低位的 1 变为 0 之后的结果。

这样我们可以利用这个位运算的性质加速我们的检查过程，在实际代码中，我们不断让当前的 n 与 n−1 做与运算，直到 n 变为 0 即可。因为每次运算会使得 n 的最低位的 1 被翻转，因此运算次数就等于 n 的二进制位中 1 的个数。

```java
		int re = 0;
        while (n != 0) {
            n &= n - 1;
            re++;
        }
        return re;
```

### [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)

### [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

与260题相同

将所有的数字异或得到sum，sum就是只出现一次的两个数的异或结果即a^b

考虑把所有数字分为两组：
1.两个只出现一次的数在不同的组中

2.相同的数字在相同的组中

注：这两个组的大小不一定相同

分组异或就得到了a和b

在异或结果中找到为1的位对应的数div，因为a^b的该位为1，所以它们的这一位不同，可以根据这一位是否为0分组，a和b就在不同的组中，而且相同的数字必定被分到一组，因为相同的数字每一位都相同，这样就满足了条件

先对所有数字进行一次异或，得到两个出现一次的数字的异或值。

在异或结果中找到任意为 1 的位。

根据这一位对所有的数字进行分组。

在每个组内进行异或操作，得到两个数字。

```java
		int sum=0;
        for(int num:nums)
        {
            sum^=num;
        }
        int div=1;
        while((div&sum)==0)
        {
            div<<=1;
        }
        int a=0;
        int b=0;
        for(int num:nums)
        {
            if((num&div)==0)
            {
                a^=num;
            }
            else{
                b^=num;
            }
        }
        return new int[]{a,b};
```



### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

如果没有进位，a与b异或就是a+b

如果有进位，进位组合起来的数就是(a&b)<<1，因此还需要将a^b与它异或，直到没有进位为止

```java
	public int add(int a, int b) {
        //无进位，用异或
        //进位用&计算
        while(b!=0)
        {
            int c=(a&b)<<1;
            a=a^b;
            b=c;
        }
        return a;
    }
```

### [318. 最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/)

怎样快速判断两个字母串是否含有重复数字呢？可以为每个字母串建立一个长度为 26 的二进制数字，每个位置表示是否存在该字母。如果两个字母串含有重复数字，那它们的二进制表示的按位与不为 0。同时，我们可以建立一个哈希表来存储字母串（在数组的位置）到二进制数字的映射关系，方便查找调用。

```java
	public int maxProduct(String[] words) {
        int n=words.length;
        int[] mask=new int[n];
        for(int i=0;i<n;i++){
            int length=words[i].length();
            for(int j=0;j<length;j++){
                mask[i]=mask[i]|(1<<(words[i].charAt(j)-'a'));
            }
        }
        int re=0;
        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                if((mask[i]&mask[j])==0){
                    re=Math.max(re,words[i].length()*words[j].length());
                }
            }
        }
        return re;
    }
```



## 快慢指针

### 141.环形链表

[141. 环形链表 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/linked-list-cycle/)

定义两个指针，一快一慢。慢指针每次只移动一步，而快指针每次移动两步。初始时，慢指针在位置 head，而快指针在位置 head.next。这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表。否则快指针将到达链表尾部，该链表不为环形链表。

注意本题中不能把slow.val==fast.val作为判断有环的条件，而是判断slow==fast，因为链表中节点的值可能相同

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

### [202. 快乐数](https://leetcode-cn.com/problems/happy-number/)

最直接想到的是可以使用set来检查是否有循环出现

如果将每个数看作链表节点，就可以转化为判断链表是否有环的问题

```java
 	public boolean isHappy(int n) {
        if(n==1){
            return true;
        }
        int slow=n;
        int fast=nextNum(n);
        while(slow!=fast){
            if(fast==1||slow==1){
                return true;
            }
            slow=nextNum(slow);
            fast=nextNum(nextNum(fast));
        }
        return false;
    }
```

每个节点指向下一个节点的值通过计算它每个位置上的数字的平方和得到

```java
	private int nextNum(int n){
        int sum=0;
        while(n>0){
            int temp=n%10;
            sum+=temp*temp;
            n=n/10;
        }
        return sum;
    }
```



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

## 单调栈

### 316.去除重复字母

[316. 去除重复字母 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/remove-duplicate-letters/)

本题和1081题相同

要求每个字母只出现一次，且返回结果的字典序最小

难点在于使得字典序最小



维护一个单调栈（不严格单调）存储字符，更新栈顶元素使得字典序最小，尽量满足从栈底到栈顶的字符递增，如果栈顶字符大于当前字符，而且字符串在当前字符之后还有与栈顶字符相同的字符，则将栈顶字符出栈，出栈后，新的栈顶字符还是需要与当前字符相比较，直到栈顶字符小于当前字符或者字符串中当前字符之后没有与栈顶字符相同的字符为止（栈空也需要考虑），这时候再将当前字符入栈

采用visited[]数组记录该字符是否加入单调栈，只有标记为false的字符才需要加入单调栈，加入之后将其标记为true

在弹出栈顶字符时，如果字符串在后面的位置上再也没有这一字符，则不能弹出栈顶字符。为此，需要记录每个字符的剩余数量，当这个值为 0 时，就不能弹出栈顶字符了，在弹出栈顶字符后，栈顶字符应该被标记为true。

字符的数量使用数组count记录，首先对字符串做预处理，遍历一遍之后获取对应字符的数量

```java
		int[] count=new int[26];
        for(int i=0;i<n;i++)
        {
            count[s.charAt(i)-'a']++;
        }
```

遍历完当前字符，不管它是否被标记为false，都应将其对应的数量减一

顺序遍历字符串，进而维护单调栈

```java
		for(int i=0;i<n;i++)
        {
            char cur=s.charAt(i);
            if(!visited[cur-'a'])
            {
                while(!stack.isEmpty()&&stack.peek()>cur&&count[stack.peek()-'a']>0)
                {
                    visited[stack.peek()-'a']=false;
                    stack.pop();
                }
                stack.push(cur);
                visited[cur-'a']=true;
            }
            count[cur-'a']--;
        }
```

遍历完字符串之后，单调栈stack中就是结果

```java
		while(!stack.isEmpty())
        {
            re.append(stack.pop());
        }
        return re.reverse().toString();
```



### 739.每日温度

[739. 每日温度 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/daily-temperatures/)

暴力法

双重遍历，对每天的气温，向后遍历数组，找到第一个比当前气温高的下标，两个下标之差就是结果

单调栈

维护一个单调栈，存储数组的下标，确保栈中的下标对应的气温是从栈底到栈顶递减的

遍历气温数组，将下标按照如下规则入栈和出栈：

当前下标为i，如果栈不为空且当前气温大于栈顶下标对应的气温，则将栈顶下标出栈为cur，且re[cur]=i-cur表示cur天需要等i-cur天之后就有更高的温度。直到栈顶的下标对应的气温大于当前气温或者栈空，将当前的气温对应的下标i入栈，否则一直进行之前的出栈操作

对于气温在这之后都不会升高的下标，它们一直在栈中，由于re[i]初始值就是0，所以不需要对其进行额外操作

```java
		int n=temperatures.length;
        int[] re=new int[n];
        Stack<Integer> stack=new Stack<Integer>();
        for(int i=0;i<n;i++)
        {
            while(!stack.isEmpty()&&temperatures[stack.peek()]<temperatures[i])
            {
                int cur=stack.pop();
                re[cur]=i-cur;
            }
            stack.push(i);
        }
        return re;
```

## 队列

### [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

考虑使用优先队列，维护一个大根堆，滑动窗口滑动时不断更新队列，队头就是当前的最大值

结果数组长度为nums.length-k+1

```java
		int n=nums.length-k+1;
        int[] re=new int[n];
```

初始时，将数组nums的前k个元素加入优先队列，第一个滑动窗口的最大值就是当前堆顶

```java
		for(int i=0;i<k;i++)
        {
            q.offer(new int[]{nums[i],i});
        }
        re[0]=q.peek()[0];
```

后面右移窗口时，就将新的元素放入队列中，堆顶即队头的元素就是堆中所有元素的最大值，但是这个最大值可能是当前窗口左边的元素，不在当前的滑动窗口中，因此，需要将这样的元素从优先队列中移除，这样的元素位置处于滑动窗口左侧，为了方便移除这样的元素，在优先队列中使用{nums[i],i}这样的数组保存元素，优先队列中元素的排序首先考虑元素值的大小（大的在前）

```java
		PriorityQueue<int[]> q= new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] pair1, int[] pair2) {
                return pair2[0] - pair1[0];
            }
        });
```

在窗口移动的过程中，需要不断删除上述的元素，注意队列不能为空的判断，然后加入新的右侧元素nums[i+k-1]到优先队列中，当前窗口中元素的最大值就是堆顶元素

```java
		for(int i=1;i<n;i++)
        {
            while(!q.isEmpty()&&q.peek()[1]<i)
            {
                q.poll();
            }
            q.offer(new int[]{nums[i+k-1],i+k-1});
            re[i]=q.peek()[0];
        }
```

### [264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

维护一个最小堆，每次将堆顶元素取出就是下一个最小的丑数，将其分别乘以2，3，5后加入队列，加入之前使用集合set去重

初始状态下队列加入1，按此规则出队列的第n个数就是第n个丑数

为防止溢出，使用long类型

```java
		PriorityQueue<Long> q=new PriorityQueue<>();
        Set<Long> set=new HashSet<>();
        int[] a=new int[]{2,3,5};
        q.offer(1L);
        set.add(1L);
        int count=1;
        long cur=1;
        while(count<=n){
            for(int i=0;count<n&&i<3;i++)
            {
                long temp=cur*a[i];
                if(set.add(temp))
                {
                    q.offer(temp);
                }
            }
            cur=q.poll();
            count++;
        }
        return (int)cur;
```



## 排序

### [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

把数组中的非负整数拼接成最小的数

首先把数组中的数转化为字符串，存储在list中

```java
		int n=nums.length;
        List<String> list=new ArrayList<String>();
        for(int i=0;i<n;i++)
        {
            list.add(String.valueOf(nums[i]));
        }
```

然后对list中的字符串排序，如果简单按照字符串大小排序，会出现[3,30]变成330而不是最小的303，因此需要重写字符串排序的规则

两个字符串的先后顺序如何考虑？题目要求的是拼接得到最小，两个字符串a和b拼接方式只有a+b和b+a两种，因此需要判断的是a+b与b+a的大小，如果a+b小，就将a放在前面，否则将b放在前面，重写的规则如下：

```java
		list.sort((a,b)->(a+b).compareTo(b+a));
```

也可以这样重写：

```java
		Collections.sort(list,new Comparator<String>(){
			@Override
			public int compare(String o1, String o2) {
				return (o1+o2).compareTo(o2+o1)>0?1:-1;
			}
		});
```

最后将排序好的list中的字符串按顺序拼接起来就是最小的字符串

```java
		StringBuilder re=new StringBuilder();
        for(int i=0;i<n;i++)
        {
            re.append(list.get(i));
        }
        return re.toString();
```

### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

三种颜色：

暴力法：直接排序，Arrays.sort(nums)

计数法：一趟扫描nums数组，记录0，1，2数量，然后，直接将数组nums按照0，1，2的顺序重构



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
        else
        {
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

### [剑指 Offer 41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

使用大根堆和小根堆，并且保证两堆size之差不大于1

约定大根堆min存放较小的一半数，小根堆max存放较大的一半数

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

### 560.和为K的子数组

[560. 和为 K 的子数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

暴力法

三重for循环，逐个遍历找到和为k的子数组数量

前缀和

使用数组sums存储数组的前缀和，sums长度为n+1，sums[0]=0

```java
		int n=nums.length;
        int[] sums=new int[n+1];
        for(int i=0;i<n;i++)
        {
            sums[i+1]=sums[i]+nums[i];
        }
```

然后对sums数组进行双重遍历，利用前缀和之差表示子数组的和，如sums[j]-sums[i]（j>i），表示从nums[i]到nums[j-1]的元素之和

```java
		int re=0;		
		for(int i=0;i<=n;i++)
        {
            for(int j=i+1;j<=n;j++)
            {
                if(sums[j]-sums[i]==k)
                {
                    re++;
                }
            }
        }
```

前缀和+哈希

使用哈希表map记录前缀和的值和对应的数量，避免了遍历寻找合适的前缀和的过程

并且在计算前缀和的过程中，从map中获取前缀和为sums[i+1]-k的个数，表明这两个前缀和之差为k，而且也确保了i+1大于map中前缀和的下标，因为只有在计算出前缀和之后才将其加入map。

map最开始要加入（0，1），表示sums[0]

```java
		map.put(0,1);
        int re=0;
        for(int i=0;i<n;i++)
        {
            sums[i+1]=sums[i]+nums[i];
            if(map.containsKey(sums[i+1]-k))
            {
                re+=map.get(sums[i+1]-k);
            }
            map.put(sums[i+1],map.getOrDefault(sums[i+1],0)+1);
        }
```



### 1744.你能在你最喜欢的那天吃到你最喜欢的糖果吗

[1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？ - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/)

计算candiesCount的前缀和sum[]

划分区间

在favoriteDayi天所吃糖果数量区间为[favoriteDayi,favoriteDayi*dailyCapi]，最少就是每天一颗，最多就是每天dayliCapi颗

要在规定的天数吃到favoriteTypei的糖果，需要吃的糖果数量区间为`[sum[favoriteTypei-1]+1,sum[favoriteTypei]]`，考虑左边界情况

这两个区间有交集，answer就为True

## 动态规划

![转化过程](https://pic.leetcode-cn.com/2cdf411d73e7f4990c63c9ff69847c146311689ebc286d3eae715fa5c53483cf-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-03-08%2010.23.03.png)

### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

dp[i]表示以s[i]结尾的最长无重复字符子串，

首先所有的dp[i]都为1，初始状态下，子串长度为1

将s[i]与它前面的dp[i-1]个字符比较，从后往前比，因为前dp[i-1]个字符是没有重复的

即从s[i-1]到s[i-dp[i-1]]就是以s[i-1]为结尾的最长无重复子串，因此将s[i]从后往前比较 ，直到相同的s[j]为止

此时，dp[i]=i-j，结果re取最大的dp[i]

```java
		int[] dp=new int[n];
        Arrays.fill(dp,1);
        int re=1;
        for(int i=1;i<n;i++)
        {
            int j=i-1;
            while(i-j<=dp[i-1]&&s.charAt(i)!=s.charAt(j))
            {
                j--;
            }
            dp[i]=i-j;
            re=Math.max(re,dp[i]);
        }
```



上面的动态规划过程，每一次都要检查s[i-dp[i-1],...,i-1]中与s[i]相同的字符，如果能够记录字符s[i]出现的位置，就可以优化动态规划的过程

因此使用数组pos[256]记录字符上次出现的位置，初始状态下pos[s[0]]=0，其余都是-1，表示其他的字符都没有出现过

从i=1开始遍历字符串s，dp[i]取dp[i-1]+1和i-pos[s[i]]中的较小值，最好的情况就是s[i]前面的dp[i-1]个字符都没有和它重复的，否则就是前dp[i-1]个字符中有与它重复的字符，且位置为pos[s[i]]

如果i-pos[s[i]]比dp[i-1]+1大，说明前i-pos[s[i]]-1个字符都不和s[i]重复，但是由于其它的字符重复，只能取dp[i-1]+1个字符，示例如下：“abcba”

由于dp[i]只与dp[i-1]相关，因此不需要数组存储dp

```java
		int[] pos=new int[256];
        Arrays.fill(pos,-1);
        pos[s.charAt(0)]=0;
        int dp=1;
        int re=1;
        for(int i=1;i<n;i++)
        {
            dp=Math.min(dp+1,i-pos[s.charAt(i)]);
            pos[s.charAt(i)]=i;
            re=Math.max(re,dp);
        }
```



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



### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

将word1转换为word2，可以进行三种操作

定义`dp[i][j]`表示将word1的前 i 个字符转换为word2的前 j 个字符需要的最少操作数

因此只需要考虑对于字符word1[i] 和word2[j] 

如果两个字符不相等，则可以将word1[i]替换为word2[j]，这样所需要的最少操作数就是`dp[i-1][j-1]+1`，也可以将word1[i]删除，所需操作数就是`dp[i-1][j]+1`，也可以向word1中插入word2[j]，所需操作数就是`dp[i][j-1]+1`，`dp[i][j]`只能取这三种情况的最小值

如果两个字符相等，则不需要替换，所需要的最少操作数就是`dp[i-1][j-1]`，不过还是需要和删除，插入的另外两种情况比较，取最小值

还需要考虑，字符串长度为0的特殊情况：

如果 i=0，word1的前0个字符，转化为word2的前j个字符，最少操作就是往word1添加word2的 j 个字符，即操作数为 j

如果 j=0，word1的前i个字符，转化为word2的前0个字符，最少操作就是删除word1的 i 个字符，即操作数为 i

从而得到状态转移方程：
$$
dp[i][j] = \begin{cases}
j &\text{i=0}\\
i &\text{j=0}\\
min(dp[i-1][j-1]+1,dp[i-1][j]+1,dp[i][j-1]+1) &\text{word1[i]!=word2[j]}\\
min(dp[i-1][j-1],dp[i-1][j]+1,dp[i][j-1]+1) &\text{word1[i]=word2[j]}\\

\end{cases}
$$
最后的答案就是`dp[m][n]`(m，n分别为word1，word2的长度)

```java
	//word1[i]和word2[j]
    //如果两个字符相等，则可以不进行替换操作，需要和进行另外两种操作的操作数比较
    //如果不相等，有三种选择，将word1[i]替换成word2[j]，插入word2[j]，删除word1[i]
    public int minDistance(String word1, String word2) {
        int m=word1.length();
        int n=word2.length();
        int[][] dp=new int[m+1][n+1]; //dp[i][j]表示word1的前i个字符转化为word2的前j个字符需要的最少操作数
        char[] a=word1.toCharArray();
        char[] b=word2.toCharArray();
        for(int i=0;i<=m;i++){//word2的前0个字符，word1需要全删除
            dp[i][0]=i;
        }
        for(int i=0;i<=n;i++){//word1的前0个字符，word1需要全插入字符
            dp[0][i]=i;
        }
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(a[i-1]==b[j-1]){
                    dp[i][j]=Math.min(Math.min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]);
                }else{
                    dp[i][j]=Math.min(Math.min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]+1);
                }
            }
        }
        return dp[m][n];
    }
```



### 91.解码方法

[91. 解码方法 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/decode-ways/)

线性dp

dp[i]表示s中前i个字符即s[0~i-1]编码方法的总数

如果第一个字符为0，那么直接返回0，无法解码

dp初始化：dp[0]=0,dp[1]=1

从第二个字符开始遍历字符串s，i=1~n-1，s[i]对应的解码方法为dp[i+1]

如果s[i]=0，则只有在s[i-1]=1或2时，能够解码，dp[i+1]=dp[i]，否则无法解码，直接返回0

如果s[i]!=0，s[i-1]=1或者s[i-1]=2且s[i]<6的情况，可以有两种解码方式，一种是将s[i]作为单个字母解码，还有一种是将s[i-1]和s[i]联合起来解码成一个字母，因此dp[i+1]=dp[i-1]+dp[i]

其余的情况都是dp[i+1]=dp[i]，将字符s[i]解码为一个字母

最终结果为dp[n]

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



### [264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

使用数组dp，dp[i]表示第i+1个丑数，第n个丑数即为dp[n-1]

初始dp[0]=1，最小的丑数是1

本题难点在于如何找到下一个最小的丑数，丑数有这样一个性质：丑数只能由丑数相乘得到，质因子就是2，3，5

因此使用index数组保存位置，表示下一个丑数是当前位置的丑数乘以对应的质因子得到的，三个质因子2，3，5对应的位置是index[0],index[1],index[2]

初始状态下，三个位置都是0

从i=1~n-1，`dp[i]=min(dp[index[0]]*2,dp[index[1]]*3,dp[index[2]]*5)`

并且，通过与质因子相乘得到最小丑数的对应丑数位置index[i]需要加一，表示往前移动，达到去重的目的

```java
		int[] dp=new int[n];
        dp[0]=1;
        int[] index=new int[3];
        for(int i=1;i<n;i++)
        {
            int a=dp[index[0]]*2;
            int b=dp[index[1]]*3;
            int c=dp[index[2]]*5;
            int next=Math.min(a,Math.min(b,c));
            if(next==a)
            {
                index[0]++;
            }
            if(next==b)
            {
                index[1]++;
            }
            if(next==c)
            {
                index[2]++;
            }
            dp[i]=next;
        }
        return dp[n-1];
```



### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

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



### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

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



本题还可以使用二分查找将时间复杂度降低为 *O*(*n* log *n*)。我们定义一个 dp 数组，其中 dp[k]存储长度为 k+1 的最长递增子序列的最后一个数字。我们遍历每一个位置 i，如果其对应的数字大于 dp 数组中所有数字的值，那么我们把它放在 dp 数组尾部，表示最长递增子序列长度加 1；如果我们发现这个数字在 dp 数组中比数字 *a* 大、比数字 *b* 小，则我们将 *b* 更新为此数字，使得之后构成递增序列的可能性增大。以这种方式维护的 dp 数组永远是递增的，因此可以用二分查找加速搜索。

```java
	public int lengthOfLIS(int[] nums) {
        int n=nums.length;
        //dp中存放的为严格递增子序列
        int[] dp=new int[n+1];
        int size=1;//size为严格递增子序列的长度
        dp[size]=nums[0];
        for(int i=1;i<n;i++){
            if(dp[size]<nums[i]){//如果nums[i]比递增子序列中最大元素还大，直接将其加入序列，并将长度size加一
                dp[++size]=nums[i];
            }else{
                int index=binarysearch(dp,size,nums[i]);//找到dp中第一个比nums[i]大的数
                dp[index]=nums[i];
            }
        }
        return size;
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

如果为偶数，考虑两个子集中的一个子集是否能达到`m=sum/2`，为方便dp，设该子集为较小的那一个，那么动态规划的目的就是让该子集的和取最大不断接近m

`dp[i][j]` 表示数组前 i 个元素中，不大于 j 的最大值，该问题转化为0-1背包问题，num表示数组nums中的第 i 个元素

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

使用一维数组dp[m+1]，为了防止上一层循环的`dp[0,.....,j-1]`被覆盖，循环的时候 j 只能逆向遍历

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



### [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

转化为0-1背包问题，有两个背包的大小，即0的数量和1的数量

```java
	//0-1背包问题
    public int findMaxForm(String[] strs, int m, int n) {
        int len=strs.length;
        int[][] count=new int[len][2];//将字符串数组中每个字符串的0和1数量保存下来
        for(int i=0;i<len;i++){
            for(char c:strs[i].toCharArray()){
                if(c=='0'){
                    count[i][0]++;
                }else{
                    count[i][1]++;
                }
            }
        }
        int[][][] dp=new int[len+1][m+1][n+1];
        for(int i=1;i<=len;i++){
            for(int j=0;j<=m;j++){
                for(int k=0;k<=n;k++){
                    int count0=count[i-1][0];
                    int count1=count[i-1][1];
                    dp[i][j][k]=dp[i-1][j][k];
                    if(j>=count0&&k>=count1){
                        dp[i][j][k]=Math.max(dp[i][j][k],1+dp[i-1][j-count0][k-count1]);
                    }
                }
            }
        }
        return dp[len][m][n];
    }
```



空间压缩优化

注意遍历顺序

```java
	public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp=new int[m+1][n+1];
        for(String s:strs){
            int count0=0,count1=0;
            for(char c:s.toCharArray()){
                if(c=='0'){
                    count0++;
                }else{
                    count1++;
                }
            }
            for(int j=m;j>=count0;j--){
                for(int k=n;k>=count1;k--){
                    dp[j][k]=Math.max(dp[j][k],1+dp[j-count0][k-count1]);
                }
            }
        }
        return dp[m][n];
    }
```





### 486.预测赢家

[486. 预测赢家 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/predict-the-winner/)

对玩家来说，他需要尽可能地将他与另一名玩家分数的差值最大化

`dp[i][j]` 表示当数组剩下的部分为下标 i 到下标 j 时，当前玩家与另一个玩家的分数之差的最大值，当前玩家不一定是先手

只有当 i<=j 时，数组剩下的部分才有意义，因此当 i>j 时，`dp[i][j]=0`

当 i=j 时，`dp[i][i]`表示只剩下第 i 个分数，那么玩家只能选择这一个，因此将`dp[i][i]`初始化为nums[i]，其余的为0

对于当前的玩家来说，如果选择最左边的数即nums[i]，差值为nums[i]减去下一个玩家的`dp[i+1][j]`，如果选择最右边的数即nums[j]，差值为nums[j]减去`dp[i][j-1]`，这是根据留给下一个玩家的数组剩余部分而定的，而当前玩家需要选择这两者中最大的

状态转移方程为：
$$
dp[i][j]=max(nums[i]-dp[i+1][j],nums[j]-dp[i][j-1]) 当i<j
$$
我们看看状态转移的方向，它指导我们填表时采取什么计算方向，才不会出现：求当前的状态时，它所依赖的状态还没求出来。

`dp[i][j]`依赖于`dp[i+1][j]和dp[i][j-1]`，因此 i 的值是从大到小，而 j 的值是从小到大，并且 j 要大于 i



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

最后的结果差值为`dp[0][n-1]`

因为`dp[i][j]`依赖于`dp[i+1][j]`和`dp[i][j-1]`，可以进行空间优化，遍历顺序照常

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

则有 `target=sum-2*neg`

从而 `neg=(sum-target)/2`，且`sum-target`必须为非负偶数（因为neg为非负整数），否则无解

将题目转化为nums数组中取任意个整数和为neg的方案数量

`dp[i][j]`表示数组nums中前 i 个整数组成的结果为 j 的方案数量，dp[n] [neg]即为答案

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
对于nums中第i个整数num，如果 j 小于num则`dp[i][j]=dp[i-1][j]`，否则`dp[i][j]=dp[i-1][j]+dp[i-1][j-num]`。

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
题解为`dp[0][n-1]`

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
题解为`dp[n][amount]`

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

因此使用`dp[i][j]`表示字符串从下标 i 开始字符`（'a'+j）`出现的第一个位置，如果没有该字符则`dp[i][j]`=m，即字符串s的长度，表示下标 i **及之后**没有该字符

`dp[m+1][26]`，初始化dp[m]的值都为m，从后往前遍历数组dp

如果字符串s的第i个字符恰好为`'a'+j`，则`dp[i][j]=i`，否则`dp[i][j]=dp[i+1][j]`，即在下标 i **之后**字符`'a'+j`第一次出现的位置，所以需要从后往前遍历数组

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



### [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

一般来说，因为这道题涉及到四个方向上的最近搜索，所以很多人的第一反应可能会是广度优先搜索。但是对于一个大小 *O*(*mn*) 的二维数组，对每个位置进行四向搜索，最坏情况的时间复杂度（即全是 1）会达到恐怖的 *O*(*m*2*n*2)。

一种办法是使用一个 dp 数组做 memoization，使得广度优先搜索不会重复遍历相同位置；

另一种更简单的方法是，我们从左上到右下进行一次动态搜索，再从右下到左上进行一次动态搜索。两次动态搜索即可完成四个方向上的查找。

### 583.两个字符串的删除操作

[583. 两个字符串的删除操作 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

本题可以在300题的基础上做

先求出最长递增子序列的长度`dp[n][m]`，然后用两个字符串的长度分别减去`dp[n][m]`，求和就是删除所需的最小步数

也可以直接就删除的最小步数进行动态规划



### 647.回文子串

[647. 回文子串 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/palindromic-substrings/)

与5题相同的解法，本题求的是回文子串的数量，所以在`dp[i][j]=true`时，计数器加一即可

### 673.最长递增子序列的个数

[673. 最长递增子序列的个数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)

本题可以直接使用300题的方法，在状态转移的过程中得到最长递增子序列的长度和对应的个数

在300题的基础上进行改造

dp[i]仍然是以nums[i]结尾的递增子序列的最大长度，定义count[i]表示以nums[i]为结尾的最长递增子序列的个数

dp[i]和count[i]的初始值都为1

状态转移是顺序遍历，先计算dp[0...i-1]的值，count[i]也类似

只有当	nums[i]>nums[j]时才需要进行更新操作

如果dp[i]<dp[j]+1；说明以 nums[i]为结尾的最长递增子序列是以nums[j]为结尾的最长递增子序列的尾部加上nums[i]组成的，因此count[i]更新为count[j]

如果dp[i]=dp[j]+1，说明以nums[i]为结尾的最长递增子序列的倒数第二个数有多个取值，因此将count[i]加上count[j]

遍历过程中也需要更新最长递增子序列的长度max和对应的数量re

如果max<dp[i]，则将max更新为dp[i]，将re更新count[i]，这种情况就是找到了更长的递增子序列

如果max=dp[i]，则max不需要变动，将re加上count[i]，这种情况对应的是最长递增子序列可以通过不同的nums[i]作为结尾得到

```java
		for(int i=0;i<n;i++)
        {
            dp[i]=1;
            count[i]=1;
            for(int j=0;j<i;j++)
            {
                if(nums[i]>nums[j])
                {
                    if(dp[i]<dp[j]+1)
                    {
                        dp[i]=dp[j]+1;
                        count[i]=count[j];
                    }
                    else if(dp[i]==dp[j]+1)
                    {
                        count[i]+=count[j];
                    }
                }
            }
            if(max<dp[i])
            {
                max=dp[i];
                re=count[i];
            }
            else if(max==dp[i])
            {
                re+=count[i];
            }
        }
```



### [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

定义数组buy[n+1]和sell[n+1]

buy[i]表示第i天买入股票的最大利润，即持有股票

sell[i]表示第i天卖出股票的最大利润，即不持有股票

这两者之间存在转化关系

定义price为第i天的股票价格，即prices[i-1]

如果第i天不持有股票，利润为sell[i]，则可能的情况有两种：

​		1.前一天手上已经没有股票了，利润即sell[i-1]

​		2.前一天持有股票的利润为buy[i-1]，然后今天将它卖出，需要扣除手续费，利润即buy[i-1]+price-fee

如果第i天持有股票，利润为buy[i]，则可能的情况有两种：

​		1.前一天也是持有股票的，利润即为buy[i-1]

​		2.前一天不持有股票的利润为sell[i-1]，然后今天买入股票，利润为sell[i-1]+price

最后的最大利润必然是第n天不持有股票sell[n]

初始状态下，没有股票的利润为0。有股票的利润为-prices[0]，即sell[0]=0，buy[0]=-prices[0]

状态转移方程为：
$$
sell[i] = \begin{cases}
0 &\text{i=0}\\
max(sell[i-1],buy[i-1]+prices[i-1]-fee) &\text{0<i<=n}\\
\end{cases}
$$

$$
buy[i] = \begin{cases}
-prices[0] &\text{i=0}\\
max(buy[i-1],sell[i-1]-prices[i-1]) &\text{0<i<=n}\\
\end{cases}
$$

具体代码：

```java
 	public int maxProfit(int[] prices, int fee) {
        int n=prices.length;
        int[] buy=new int[n+1];
        int[] sell=new int[n+1];
        sell[0]=0;
        buy[0]=-prices[0];
        for(int i=1;i<=n;i++){
            sell[i]=Math.max(sell[i-1],buy[i-1]+prices[i-1]-fee);
            buy[i]=Math.max(buy[i-1],sell[i-1]-prices[i-1]);
        }
        return sell[n];
    }
```

考虑空间优化，sell[i]和buy[i]只与sell[i-1]、buy[i-1]有关，因此可以使用变量代替数组

由于可以无限次交易，所以可以在当天买入卖出，buy和sell的先后顺序不影响结果

```java
	public int maxProfit(int[] prices, int fee) {
        int n=prices.length;
        int sell=0;
        int buy=-prices[0];
        for(int i=0;i<n;i++){
            sell=Math.max(sell,buy+prices[i]-fee);
            buy=Math.max(buy,sell-prices[i]);
        }
        return sell;
    }
```



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

初始化值为`dp[0][0][0]=1`，表示至少产生利润为0的方案有一种，就是不派遣任何工人

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

回溯法（backtracking）是优先搜索的一种特殊情况，又称为试探法，常用于需要记录节点状

态的深度优先搜索。通常来说，排列、组合、选择类问题使用回溯法比较方便。

顾名思义，回溯法的核心是回溯。在搜索到某一节点的时候，如果我们发现目前的节点（及

其子节点）并不是需求目标时，我们回退到原来的节点继续搜索，并且把在目前节点修改的状态 

还原。这样的好处是我们可以始终只对图的总状态进行修改，而非每次遍历时新建一个图来储存

状态。在具体的写法上，它与普通的深度优先搜索一样，都有 [修改当前节点状态]→[递归子节

点] 的步骤，只是多了回溯的步骤，变成了 [修改当前节点状态]→[递归子节点]→[回改当前节点

状态]。

没有接触过回溯法的读者可能会不明白我在讲什么，这也完全正常，希望以下几道题可以让

您理解回溯法。如果还是不明白，可以记住两个小诀窍，一是按引用传状态，二是所有的状态修 

改在递归完成后回改。

回溯法修改一般有两种情况，一种是修改最后一位输出，比如排列组合；一种是修改访问标

记，比如矩阵里搜字符串。

### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

在二叉搜索树中获得递增序列就要使用中序遍历

要转换成排序的循环双向链表，使用head指向最小的节点，在递归遍历的过程中使用pre指向当前节点的前驱

最开始pre为空，根据中序遍历，一直遍历到二叉搜索树最左边的节点，也就是最小的节点，将head指向它，判断到达最左边是根据pre==null。处理完最左边的节点后，将pre指向该节点，然后遍历其右子树，这个过程中需要将node的left指向pre，pre.right指向pre并跟随着中序遍历不断更新pre，

遍历完整个二叉搜索树之后，head指向最小的节点，也就是循环双向链表的头部，pre指向最大的节点，即链表的尾部，由于是循环双向链表，所以需要将头尾双向连接，即head.left=pre，pre.right=head

```java
	private Node head;
    private Node pre;
    public Node treeToDoublyList(Node root) {
        if(root==null)
        {
            return head;
        }
        pre=null;
        track(root);
        //首尾相连
        head.left=pre;
        pre.right=head;
        return head;
    }
    private void track(Node node)
    {
        if(node==null)
        {
            return;
        }
        track(node.left);
        if(pre==null)//到二叉搜索树的最左边
        {
            head=node;
        }
        else
        {
            pre.right=node;//建立后继
        }
        node.left=pre;//建立前驱
        pre=node;
        track(node.right);
    }
```



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



### [301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)

对括号的判断，以一个score来记录，遍历字符串，如果是左括号就加一，如果是右括号就减一，其他字符不操作，因此字符串括号无效的情况就是score小于0，或者score过高（左括号太多）

如何判断score过高？

首先遍历字符串，获得左括号的数量l和右括号的数量r，score能取到的最大值max取l和r中的较小值

在递归的过程中，如果score>max就说明此时score过高，括号无效

```java
		if(score<0||score>max)
        {
            return;
        }
```

字符串temp合法的条件是已经遍历完了字符串s，且score=0，说明字符串中所有的括号有效

为了避免重复，使用set去重，递归过程中将合法字符串加入set

要求删除最小数量的无效括号，则添加到结果中的合法字符串长度都一样，且是合法字符串中最长的，使用len记录当前最长合法字符串的长度，因此 set中的所有字符串长度都等于len

在递归过程中，只有当合法字符串长度不小于len时，才考虑将其加入set，并且，如果该合法字符串长度大于len，说明这时set中的所有字符串都不满足删除最小数量的五小括号这个条件，因此将set清空，将len更新为更长的该合法字符串的长度，将其加入set

```java
		if(start==s.length())
        {
            if(score==0&&temp.length()>=len)
            {
                if(temp.length()>len)
                {
                    set.clear();
                    len=temp.length();
                }
                set.add(temp);
            }
            return;
        }
```

递归方法是对于当前位置的字符

如果是左括号或者右括号，可以选择将其加入字符串temp或者不加，如果加入score就要相应的加一减一，否则不用

如果是其他字符，则直接加入temp

每一步递归都将当前位置后移一位，即start+1

```java
	Set<String> set=new HashSet<>();
    int len=0;
    int max=0;
    public List<String> removeInvalidParentheses(String s) {
        int l=0,r=0;
        for(char c:s.toCharArray())
        {
            if(c=='(')
            {
                l++;
            }
            else if(c==')')
            {
                r++;
            }
        }
        max=Math.min(l,r);
        dfs(s,new String(),0,0);
        return new ArrayList<>(set);
    }
    private void dfs(String s,String temp,int start,int score)
    {
        if(score<0||score>max)
        {
            return;
        }
        if(start==s.length())
        {
            if(score==0&&temp.length()>=len)
            {
                if(temp.length()>len)
                {
                    set.clear();
                    len=temp.length();
                }
                set.add(temp);
            }
            return;
        }
        char c=s.charAt(start);
        if(c=='(')
        {
            dfs(s,temp+String.valueOf(c),start+1,score+1);
            dfs(s,temp,start+1,score);
        }
        else if(c==')')
        {
            dfs(s,temp+String.valueOf(c),start+1,score-1);
            dfs(s,temp,start+1,score);
        }
        else{
            dfs(s,temp+String.valueOf(c),start+1,score);
        }
    }
```

剪枝策略

首先确定最长合法字符串的长度len

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

## BFS

广度优先遍历

广度优先搜索（breadth-fifirst search，BFS）不同与深度优先搜索，它是一层层进行遍历的，因

此需要用先入先出的队列而非先入后出的栈进行遍历。由于是按层次进行遍历，广度优先搜索时

按照“广”的方向进行遍历的，也常常用来处理最短路径等问题。

这里要注意，深度优先搜索和广度优先搜索都可以处理可达性问题，即从一个节点开始是否

能达到另一个节点。因为深度优先搜索可以利用递归快速实现，很多人会习惯使用深度优先搜索

刷此类题目。实际软件工程中，笔者很少见到递归的写法，因为一方面难以理解，另一方面可能

产生栈溢出的情况；而用栈实现的深度优先搜索和用队列实现的广度优先搜索在写法上并没有太

大差异，因此使用哪一种搜索方式需要根据实际的功能需求来判断。

### [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

利用队列实现BFS

首先将根节点入队列q，如果根节点root为空，则直接返回空数组

对于队列中的每一个节点，出队后，将其值加入结果list，再将其不为空的左右子节点加入队列

```java
		q.offer(root);
        while(!q.isEmpty())
        {
            TreeNode node=q.poll();
            list.add(node.val);
            if(node.left!=null)
            {
                q.offer(node.left);
            }
            if(node.right!=null)
            {
                q.offer(node.right);
            }
        }
```

最后将list遍历转化为数组

### [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

和32-I类似，增加层次

每一层就是队列当前的所有节点，所以首先将队列长度size保留，然后将队列中的前size个节点按照32-I中的方式处理

```java
		q.offer(root);
        while(!q.isEmpty())
        {
            int size=q.size();
            list.add(new ArrayList<Integer>());
            while(size>0)
            {
                TreeNode node=q.poll();
                list.get(level).add(node.val);
                if(node.left!=null)
                {
                    q.offer(node.left);
                }
                if(node.right!=null)
                {
                    q.offer(node.right);
                }
                size--;
            }
            level++;
        }
```

### [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

在32-II的基础上添加对层次的判断，决定是逆序还是顺序添加数值

```java
		q.offer(root);
        while(!q.isEmpty())
        {
            int size=q.size();
            List<Integer> list=new ArrayList<>();
            while(size>0)
            {
                TreeNode node=q.poll();
                if((level&1)==0)
                {
                    list.add(node.val);//顺序
                }
                else
                {
                    list.add(0,node.val);//逆序
                }
                if(node.left!=null)
                {
                    q.offer(node.left);
                }
                if(node.right!=null)
                {
                    q.offer(node.right);
                }
                size--;
            }
            re.add(list);
            level++;
        }
```



### [310. 最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)

首先，我们看了样例，发现这个树并不是二叉树，是多叉树。
然后，我们可能想到的解法是：根据题目的意思，就挨个节点遍历bfs，统计下每个节点的高度，然后用map存储起来，后面查询这个高度的集合里最小的就可以了。
但是经过尝试这样会超时。
于是我们看图（题目介绍里面的图）分析一下，发现，越是靠里面的节点越有可能是最小高度树。
所以，我们可以这样想，我们可以倒着来。
我们从边缘开始，先找到所有出度为1的节点，也就是叶子节点，然后把所有出度为1的节点进队列，然后不断地bfs，最后找到的就是两边同时向中间靠近的节点，那么这个中间节点就相当于把整个距离二分了，那么它当然就是到两边距离最小的点，也就是到其他叶子节点最近的节点了。

得到的解决方案就是寻找最中间的节点，一层一层地删除叶子节点，最后一层就是答案、

```java
	//最中间的节点就是答案，一层一层地删除叶子节点
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer>[] arr=new ArrayList[n];//保存与当前节点的邻接节点
        Queue<Integer> q=new LinkedList<>();
        List<Integer> re=new ArrayList<>();
        if(n==1){//只有一个节点，直接返回0，面向测试用例编程
            re.add(0);
            return re;
        }
        int[] degree=new int[n];//保存节点的度
        for(int i=0;i<n;i++){
            arr[i]=new ArrayList<Integer>();
        }
        for(int[] edge:edges){//根据边，得到节点互通的关系
            arr[edge[0]].add(edge[1]);
            arr[edge[1]].add(edge[0]);
            degree[edge[0]]++;
            degree[edge[1]]++;
        }
        for(int i=0;i<n;i++){
            if(degree[i]==1){
                q.offer(i);//将度为1的节点即叶子节点入队列
            }
        }
        while(!q.isEmpty()){
            re=new ArrayList<>();//每一轮re都会更新
            int size=q.size();
            while(size>0){
                size--;
                int cur=q.poll();
                re.add(cur);//re每一轮bfs都会更新，最后一轮bfs到的叶子节点为答案
                for(int i=0;i<arr[cur].size();i++){
                    int temp=arr[cur].get(i);
                    degree[temp]--;//将当前节点的邻接节点度减一，即相当于把当前节点cur删除
                    if(degree[temp]==1){
                        q.offer(temp);//将度为1 的节点即叶子节点入队列
                    }
                }
            }
        }
        return re;//此时的re为最中间的一层叶子节点，就是答案
    }
```



### [934. 最短的桥](https://leetcode-cn.com/problems/shortest-bridge/)

先确定第一个岛的边界，可以通过dfs实现，把第一个岛都变成-1

从边界开始bfs，一圈一圈往外扩张，直到与另一个岛相邻，扩张的圈数就是答案

用队列实现bfs，具体的，在dfs确定第一个岛的范围时，将第一个岛边界的外围水坐标入队列

```java
	private void dfs(int[][] grid,Queue<int[]>q,int i,int j){
        if(i<0||i>=n||j<0||j>=n||grid[i][j]==-1){
            return;
        }
        if(grid[i][j]==0)
        {
            q.offer(new int[]{i,j});//第一个岛边界外的点进入队列，便于后续bfs
            return;
        }
        grid[i][j]=-1;
        for(int k=0;k<4;k++){
            dfs(grid,q,i+move[k],j+move[k+1]);
        }
    }
```



之后将队列中的水坐标出队列，并对其上下左右的坐标进行判断：

如果是0，则是水，将其入队列，并置为-1，表明扩张

如果是-1，则是第一座岛的范围，不做处理

如果是1，则是第二座岛，扩张结束，返回圈数level

```java
		//bfs一圈一圈地扩张
        int level=0;//扩张的圈数
        while(!q.isEmpty()){
            level++;//由于队列中是第一座岛的外围水坐标，相当于已经扩张了一圈，因此圈数level初始就需要加1，
            int size=q.size();
            while(size>0){
                size--;
                int[] cur=q.poll();
                for(int k=0;k<4;k++){
                    int x=cur[0]+move[k];
                    int y=cur[1]+move[k+1];
                    if(x>=0&&x<n&&y>=0&&y<n){//防止越界
                        if(grid[x][y]==0){//是水，扩张
                            grid[x][y]=-1;
                            q.offer(new int[]{x,y});
                        }else if(grid[x][y]==1){//到了下一个岛
                            return level;
                        }
                    }
                }
            }
        }
```



## DFS

深度优先遍历

### [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

深度优先遍历，使用层次level确定将数值添加到哪一层

遍历子节点，层次level加一 

需要注意的是，如果当前层次还没有初始化，先进行初始化

```java
	private void dfs(TreeNode root,int level)
    {
        if(root==null)
        {
            return;
        }
        if(list.size()<level+1)
        {
            list.add(new ArrayList<Integer>());
        }
        list.get(level).add(root.val);
        dfs(root.left,level+1);
        dfs(root.right,level+1);
    }
```



### [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

和32-II类似，添加了层次判断，决定是逆序还是顺序

```java
	private void dfs(TreeNode root,int level)
    {
        if(root==null)
        {
            return;
        }
        if(list.size()<level+1)
        {
            list.add(new ArrayList<Integer>());
        }
        if((level&1)==0)
        {
            list.get(level).add(root.val);//顺序
        }
        else
        {
            list.get(level).add(0,root.val);//逆序
        }
        dfs(root.left,level+1);
        dfs(root.right,level+1);
    }
```

### [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

可以不考虑LeetCode使用的序列化方法，而是使用先序遍历，并且将节点的空左右节点补全，如果节点为空则字符串加入"null"，并以”,“分割每个节点的值，序列化过程如下：

```java
	public String serialize(TreeNode root) {
        StringBuilder sb=new StringBuilder();
        if(root==null)
        {
            return "null,";
        }
        else{
            sb.append(root.val+",");
        }
        sb.append(serialize(root.left));
        sb.append(serialize(root.right));
        return sb.toString();
    }
```



反序列化首先将字符串以”,“分割为数组，并构造列表list，

```java
	public TreeNode deserialize(String data) {
        //TreeNode root=null;
        String[] arr=data.split(",");
        List<String> list=new ArrayList<>();
        for(String s:arr)
        {
            list.add(s);
        }
        return dfs(list);
    }

```

使用dfs构造二叉树，构造过程中，如果是”null“，表明该节点为空，直接返回，空节点没有子节点

否则不为空，将值取出并生成新节点，每次都是取列表的第一个元素，因此每次取出后都需要删除列表的第一个元素，来表示构造的进度

```java
	private TreeNode dfs(List<String> list)
    {
        String cur=list.get(0);
        list.remove(0);
        if(cur.equals("null"))
        {
            return null;
        }
        TreeNode node=new TreeNode(Integer.parseInt(cur));
        node.left=dfs(list);
        node.right=dfs(list);
        return node;
    }
```



### 200.岛屿数量

[200. 岛屿数量 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/number-of-islands/)

将二维网格看成一个无向图，竖直或水平相邻的 1 之间有边相连。

为了求出岛屿的数量，我们可以扫描整个二维网格。如果一个位置为 1，则以其为起始节点开始进行深度优先搜索。在深度优先搜索的过程中，每个搜索到的 1都会被重新标记为 0，这样就可以唯一确定每一个岛屿。

最终岛屿的数量就是我们进行深度优先搜索的次数。

### [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

如果一个点一个点的判断能否流入大西洋，太平洋，那么深度遍历将会非常复杂

逆向思维，看边界能与哪些点连接，左上为一组，右下为一组，然后两组取公共点就是答案

点的上下左右移动可以取巧：

```java
int[] move=new int[]{-1,0,1,0,-1};
```

对于点(i,j)，上下左右移动后的点就是（i+move[k],j+move[k+1]）k=0~3



```java
		m=heights.length;
        n=heights[0].length;
        boolean[][] reachP=new boolean[m][n];//记录能否到太平洋
        boolean[][] reachA=new boolean[m][n];//记录能否到大西洋
```



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



## 分治

### [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

对于区间[start,end] ，不同的二叉搜索树的区别就是根节点，以根节点 i 划分左右子树的节点

左子树的节点就是[start,i-1]

右子树的节点就是[i+1,end]

当start大于end时，节点为空

i遍历start到end，将i作为根节点，求得左右子树的节点列表，遍历左右子树列表，连接构造二叉搜索树

为了减少时间消耗，使用memo暂存结果

```java
	List<TreeNode>[][] memo;//记忆化搜索
    public List<TreeNode> generateTrees(int n) {
        memo=new List[n+1][n+1];
        return separate(1,n);
    }
    private List<TreeNode> separate(int start,int end){
        List<TreeNode> re=new ArrayList<>();
        if(start>end){
            re.add(null);
            return re;
        }
        if(memo[start][end]!=null){
            return memo[start][end];
        }
        for(int i=start;i<=end;i++){
            List<TreeNode> left=separate(start,i-1);
            List<TreeNode> right=separate(i+1,end);
            for(TreeNode leftNode:left){
                for(TreeNode rightNode:right){
                    TreeNode node=new TreeNode(i);
                    node.left=leftNode;
                    node.right=rightNode;
                    re.add(node);
                }
            }
        }
        memo[start][end]=re;
        return re;
    }
```



### [241. 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/)

以运算符为界限分割字符串的左右两边，直到字符串中没有运算符只有数字为止，然后将左右两边运算得到结果

```java
	public List<Integer> diffWaysToCompute(String expression) {
        //以运算符分割字符串左右两边，递归得到不同括号的运算结果
        //最后归总
        List<Integer> re=new ArrayList<>();
        for(int i=0;i<expression.length();i++){
            char c=expression.charAt(i);
            if(c=='+'||c=='-'||c=='*'){
                List<Integer> left=diffWaysToCompute(expression.substring(0,i));
                List<Integer> right=diffWaysToCompute(expression.substring(i+1,expression.length()));
                for(Integer leftNum:left){
                    for(Integer rightNum:right){
                        if(c=='+'){
                            re.add(leftNum+rightNum);
                        }else if(c=='-'){
                            re.add(leftNum-rightNum);
                        }else if(c=='*'){
                            re.add(leftNum*rightNum);
                        }
                    }
                }
            }
        }
        if(re.size()==0)//没有运算符，只有数字
        {
            re.add(Integer.valueOf(expression));
        }
        return re;
    }
```



也可以将中间字符串的运算结果保存起来，减少时间消耗

## 数学

### [172. 阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes/)

暴力法需要用到BigInteger

考虑结果中尾随零是怎么产生的，尾随零都是阶乘中出现了10=2*5的乘数因子

因此找阶乘每一个元素的质因子中2和5的数量，就可以得到尾随零的数量

由于2的数量远大于5的数量，因此只需要遍历1到n统计质因子为5的数量

```java
	private int find(int n,int mod){
        int re=0;
        while(n>=mod){
            if(n%mod==0){
                re++;
            }else{
                break;
            }
            n=n/mod;
        }
        return re;
    }
```



改进之后

```java
return n==0?0:n/5+trailingZeroes(n/5);
```



### [204. 计数质数](https://leetcode-cn.com/problems/count-primes/)

如果从1开始一个个遍历到n ，判断每个数是不是质数，会超时

因此使用埃拉托斯特尼筛法（Sieve of Eratosthenes，简称埃氏筛法），它是非常常用的，判断一个整数是否是质数的方法。并且它可以在判断一个整数 *n* 时，同时判断所小于 *n* 的整数，因此非常适合这道题。其原理也十分易懂：从 1 到 *n* 遍历，假设当前遍历到 *m*，则把所有小于 *n* 的、且是 *m* 的倍数的整数标为和数；遍历完成后，没有被标为和数的数字即为质数。

```java
	public int countPrimes(int n) {
        int re=0;
        boolean[] judge=new boolean[n];
        for(int i=2;i<n;i++)
        {
            if(!judge[i])
            {
                re++;
            }
            if((long)i*i<n)
            {
                for(int j=i*i;j<n;j+=i)
                {
                    judge[j]=true;
                }
            }
        }
        return re;
    }
```



### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

不使用除法

对于nums[i]，数组中除了它之外的乘积就是nums[0]~nums[i]的乘积，乘以nums[i+1]~nums[n-1]的乘积

即该元素的左右两边元素的乘积，因此利用前缀和的思想，定义前缀乘积和后缀乘积

定义数组left[i]表示，nums[i]左边所有元素的乘积

定义数组right[i]表示，nums[i]右边所有元素的乘积，空间复杂度可优化

两趟遍历就可以得到答案

```java
	public int[] productExceptSelf(int[] nums) {
        int n=nums.length;
        int[] left=new int[n];
        left[0]=1;
        for(int i=1;i<n;i++)
        {
            left[i]=nums[i-1]*left[i-1];
        }
        int right=1;
        for(int i=n-1;i>=0;i--)
        {
            left[i]=left[i]*right;
            right*=nums[i];
        }
        return left;  
    }
```



### [462. 最少移动次数使数组元素相等 II](https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements-ii/)

数组中每个数应该变成多少呢，经过数学证明是中位数

证明如下：

很多人不明白为什么中位数是最优解，其实证明非常简单，下面来看看吧：

为了方便，我们先假设一共有2n+1个数，它们从小到大排序之后如下：

```stylus
 . . . a m b . . .
```

其中m是中位数。此时，m左边有n个数，m右边也有n个数。我们假设把m左边所有数变成m需要的代价是x，把m右边所有数变成m的代价是y，此时的总代价就是`t = x+y`

好，如果你觉得中位数不是最优解，我们来看看把所有数都变成a的总代价是多少。 由于之前m右边n个数变成m的代价是y，现在让右边的数全变成a，此时右边的数的代价是`y+(m-a)*n`；m左边的n个数全变成a，它们的代价会减少到`x-(m-a)*n`。所以两边相加，结果还是 `x-(m-a)*n + y+(m-a)*n == x+y`。 但是，别忘了，m也要变成a，所以总代价是x+y+m-a，大于x+y。同理，如果让所有数都变成比m大的b，总代价则变为x+y+b-m（你可以自己算一下），依然比x+y大。并且越往左移或者往右移，这个值都会越来越大。 因此，在有2n+1个数的时候，选择中位数就是最优解。

这个证明同样可以很简单地推广到2n个数。

```stylus
. . . a b . . .
```

假设a左边有n-1个数，b右边也有n-1个数。如果我们选择把所有数都变成a，设a左边所有数变成a的代价是x，b右边所有数变成a的代价是y，因此总代价是`x+y+b-a`（b也要变成a）。 现在尝试下如果把所有数变成b，那么a左边的总代价变成了`x+(b-a)*(n-1)`，b右边总代价变成了`y-(b-a)*(n-1)`，同时还要把a变成b，因此总代价同样为`x+(b-a)*(n-1)+y-(b-a)*(n-1) == x+y+b-a`。也就是说当总个数为2n时，两个中位数选哪个最终结果都一样，但是继续左移和继续右移，都会使总代价增加（可以自己试试）。

至此，证明了`中位数是最优策略`

### 随机与取样

### [382. 链表随机节点](https://leetcode-cn.com/problems/linked-list-random-node/)

**蓄水池抽样算法**

当内存无法加载全部数据时，如何从包含未知大小的数据流中随机选取k个数据，并且要保证每个数据被抽取到的概率相等。

本题的情况就是k=1

假设数据流有N个数，要保证所有数被选中的概率相等，也就是说每个数被选中的概率都是1/N

方案为：遍历数据流时，每次只保留一个数，当遇到第i个数时，以1/i的概率保留它，以(i-1)/i的概率保留原来的数，直到把第N个数遍历完，最后保留的数就是结果

![1.png](https://pic.leetcode-cn.com/831bdf1ea840c47b79007f206fb9fe6f1a1effb6c5ceed15509fe0abb23ed2f9.jpg)

对于本题的情况来说，顺序遍历链表，按照上述k=1的方案保留下来的就是答案

```java
	public int getRandom() {
        int i=1;
        ListNode node=head;
        int re=node.val;
        while(node!=null){
            int judge=rand.nextInt(i)+1;//judge从1~i中随机生成
            if(judge==1){//只有当judge等于1时，才将re替换为当前节点值，即1/i的概率使用第i个节点值
                re=node.val;
            }
            i++;
            node=node.next;
        }
        return re;
    }
```



### [528. 按权重随机选择](https://leetcode-cn.com/problems/random-pick-with-weight/)

生成权重的前缀和数组sum

随机生成1到sum[n-1]的整数x，在sum数组中查找第一个大于或等于x的下标，就是答案

sum是前缀和数组，单调递增，所以可以使用二分查找

```java
	public int pickIndex() {
        int x=rand.nextInt(sum[n-1])+1;//生成的随机数是1~sum[n-1]，闭区间
        //结合二分查找，找sum中到最小的大于或等于x的下标
        int left=0;
        int right=n-1;
        while(left<=right){
            int mid=(left+right)/2;
            if(sum[mid]==x){
                return mid;
            }else if(sum[mid]>x){
                right=mid-1;
            }else if(sum[mid]<x){
                left=mid+1;
            }
        }
        return left;
    }
```



## 图

### 二分图

### [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/)

可以采用染色的方法，同样颜色的节点就在二分图的同一子集

规定每条边的两个节点在不同子集可以确定，每个节点和其相邻节点都应被标记为不同的颜色

根据这样的规则可以使用深度优先搜索或广度优先搜索来遍历节点并染色

因此如果发现节点与相邻节点颜色相同，就可以判定不是二分图

BFS，使用队列保存已经访问过的节点

```java
	//广度优先搜索
    public boolean isBipartite(int[][] graph) {
        int n=graph.length;
        int[] color=new int[n];//0表示节点未访问，1和2用来区别节点在分割后的哪一个子集
        Queue<Integer> q=new LinkedList<>();
        
        for(int i=0;i<n;i++){
            if(color[i]==0){//将未访问过的节点标记为1
                color[i]=1;
                q.offer(i);
            }
            while(!q.isEmpty()){
                int cur=q.poll();
                //当前节点相邻的节点标记为与cur不同的颜色
                for(int j:graph[cur]){
                    if(color[j]==0){
                        color[j]=color[cur]==1?2:1;
                        q.offer(j);
                    }else if(color[j]==color[cur]){//如果与cur相邻的节点和cur的颜色相同，则不能二分图
                        return false;
                    }
                    
                }
            }
        }
        return true;
    }
```



深度优先搜索也可实现，染色的规则不变

```java
	int n;
    int[] color;
    boolean flag;
    public boolean isBipartite(int[][] graph) {
        n=graph.length;
        flag=true;
        color=new int[n];
        for(int i=0;i<n&&flag;i++){
            if(color[i]==0){
                dfs(graph,i,1);
            }
        }
        return flag;
    }
    private void dfs(int[][] graph,int cur,int mark){
        color[cur]=mark;
        int nextmark=mark==1?2:1;//与当前节点相邻节点应该被染成不同的颜色
        for(int j:graph[cur]){
            if(color[j]==0){
                dfs(graph,j,nextmark);
            }else if(color[j]==color[cur]){//当前节点cur与相邻节点j颜色相同，无法二分图
                flag=false;
                return;
            }
        }
    }
```



### [剑指 Offer II 106. 二分图](https://leetcode-cn.com/problems/vEAB3K/)

与785题相同

### 拓扑排序

### [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

能否完成所有课程的学习，其实就是判断有向图中是否有环

将每一门课看成一个节点；

如果想要学习课程 A之前必须完成课程 B，那么我们从 B到 A 连接一条有向边。这样以来，在拓扑排序中，B 一定出现在 A 的前面。

拓扑排序+深度优先搜索

对于一个节点 u，如果它的所有相邻节点都已经搜索完成，那么在搜索回溯到 u 的时候，u 本身也会变成一个已经搜索完成的节点。这里的「相邻节点」指的是从 u 出发通过一条有向边可以到达的所有节点。

假设我们当前搜索到了节点 u，如果它的所有相邻节点都已经搜索完成，那么这些节点都已经在栈中了，此时我们就可以把 u 入栈。可以发现，如果我们从栈顶往栈底的顺序看，由于 u 处于栈顶的位置，那么 u 出现在所有 u 的相邻节点的前面。因此对于 u 这个节点而言，它是满足拓扑排序的要求的。

这样以来，我们对图进行一遍深度优先搜索。当每个节点进行回溯的时候，我们把该节点放入栈中。最终从栈顶到栈底的序列就是一种拓扑排序。

对于图中的任意一个节点，它在搜索的过程中有三种状态，即：

「未搜索」：我们还没有搜索到这个节点；

「搜索中」：我们搜索过这个节点，但还没有回溯到该节点，即该节点还没有入栈，还有相邻的节点没有搜索完成）；

「已完成」：我们搜索过并且回溯过这个节点，即该节点已经入栈，并且所有该节点的相邻节点都出现在栈的更底部的位置，满足拓扑排序的要求。

通过上述的三种状态，我们就可以给出使用深度优先搜索得到拓扑排序的算法流程，在每一轮的搜索搜索开始时，我们任取一个「未搜索」的节点开始进行深度优先搜索。

我们将当前搜索的节点 u 标记为「搜索中」，遍历该节点的每一个相邻节点 v：

如果 v 为「未搜索」，那么我们开始搜索 v，待搜索完成回溯到 u；

如果 v 为「搜索中」，那么我们就找到了图中的一个环，因此是不存在拓扑排序的；

如果 v 为「已完成」，那么说明 v 已经在栈中了，而 u 还不在栈中，因此 u 无论何时入栈都不会影响到 (u, v)(u,v) 之前的拓扑关系，以及不用进行任何操作。

当 u 的所有相邻节点都为「已完成」时，我们将 u 放入栈中，并将其标记为「已完成」。

在整个深度优先搜索的过程结束后，如果我们没有找到图中的环，那么栈中存储这所有的 n 个节点，从栈顶到栈底的顺序即为一种拓扑排序。

本题不需要得到排序结果，因此省去对应的栈

```java
	List<List<Integer>> edges=new ArrayList<>();
    boolean re;
    int[] visited;//0表示未搜索，1表示搜索中，2表示已经完成搜索
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        visited=new int[numCourses];
        re=true;
        for(int i=0;i<numCourses;i++){
            edges.add(new ArrayList<Integer>());
        }
        for(int[] pre:prerequisites){
            edges.get(pre[1]).add(pre[0]);
        }
        for(int i=0;i<numCourses&&re;i++){
            if(visited[i]==0){
                dfs(i);
            }
        }
        return re;
    }
    private void dfs(int cur){
        visited[cur]=1;//进入搜索中的状态
        for(int v:edges.get(cur)){
            if(visited[v]==1){//如果当前节点指向的节点在搜索中，说明是有环图，必定不能拓扑排序
                re=false;
                return;
            }
            if(visited[v]==0){
                dfs(v);
                if(!re){
                    return;
                }
            }
        }
        visited[cur]=2;//完成搜索
    }
```



### [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

拓扑排序+广度优先搜索

首先构造图的邻接矩阵，为了便于理解，定义课程i指向课程j表示，必须先修完课程i才能修课程j

因此题目给出的`prerequisites[i] = [ai, bi]`，在邻接矩阵里表示的是bi指向ai

为了压缩空间，这里使用`List<List<Integer>> edges`表示图中有向边，而不需要使用邻接矩阵

将入度为0的节点入队，节点出队的时候，将其指向的其他节点的入度减一，节点出队的顺序就是课程表

```java
	public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] countIn=new int[numCourses];//记录节点入度
        List<List<Integer>> edges=new ArrayList<>();//图的有向边，edges.get(i)表示i指向的节点
        for(int i=0;i<numCourses;i++){
            edges.add(new ArrayList<Integer>());
        }
        for(int[] pre:prerequisites){
            countIn[pre[0]]++;
            edges.get(pre[1]).add(pre[0]);//pre[1]指向pre[0]
        }
        Queue<Integer> q=new LinkedList<>();
        int[] re=new int[numCourses];
        int index=0;
        //先将入度为0的节点入队
        for(int i=0;i<numCourses;i++){
            if(countIn[i]==0){
                q.offer(i);
            }
        }
        while(!q.isEmpty()){
            int size=q.size();
            while(size>0){
                int cur=q.poll();
                re[index++]=cur;//将当前节点加入结果数组
                for(int i:edges.get(cur)){
                    countIn[i]--;//将当前节点指向的节点入度减一
                    if(countIn[i]==0){//如果被指向的节点i入度为0，将节点i加入队列
                        q.offer(i);
                    }
                }
                size--;
            }
        }
        //还需要检查是否所有节点入度都为0，如果有不为0的，说明无法完成所有课程，返回空数组
        if(index!=numCourses){
            return new int[]{};
        }
        return re;
    } 
```



拓扑排序+深度优先搜索

也可以解决，需要增加一个栈来存储节点顺序，也可以用下标从右边开始的数组模拟栈

### [剑指 Offer II 113. 课程顺序](https://leetcode-cn.com/problems/QA2IGt/)

和210题相同

## 哈希表

### 128.最长连续序列

[128. 最长连续序列 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

使用集合set去重 

对于一个连续序列，按照从小到大的顺序排列，开头的元素为x，则x-1一定不在集合set中，因此遍历集合寻找开头元素x，然后在集合set中继续寻找序列后面的元素，即x+1，x+2，x+3，......，从而得到序列长度，结果取序列长度的最长值

### [855. 考场就座](https://leetcode-cn.com/problems/exam-room/)

使用TreeSet，排序集合保存已经坐人的座位，利用TreeSet自带的方法完成

```java
	TreeSet<Integer> set=new TreeSet<>();
    int n;
    public ExamRoom(int n) {
        this.n=n;
    }
    
    public int seat() {
        //注意边界情况0和n-1
        //离他最近的人，相当于在当前已经坐人的相邻两个座位之间寻找最大的间距，否则就往后安排
        int re=0;//结果座位初始化为0
        if(set.size()==0){//集合为空，说明考场里没有人，直接坐在0号座位上
            set.add(re);
            return re;
        }
        int maxdistance=set.first();//初始情况下，最大距离为set中第一个元素，因为如果座位0没人坐，那么第一个元素就是它与0的距离。如果0有人坐，
        int pre=set.first();//前一个座位
        for(Integer i:set){//遍历集合
            int distance=(i-pre)/2;//当前坐人座位与前一个坐人座位的距离
            if(distance>maxdistance){//更新最大距离的同时更新结果
                maxdistance=distance;
                re=pre+distance;
            }
            pre=i;
        }
        if(n-1-set.last()>maxdistance){//边界情况n-1，如果n-1与集合中最后一个座位的距离大于之前遍历集合得到的最大距离，则将n-1置为答案
            re=n-1;
        }
        set.add(re);
        return re;
    }
    
    public void leave(int p) {
        set.remove(p);
    }
```



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

## 字典树/前缀树

字典树又称前缀树，模板为：

```java
class TrieTree{
    class TrieNode{
        TrieNode[] links=new TrieNode[26];
        String word;
        private boolean isEnd;
        public boolean isEnd()
        {
            return isEnd;
        }
    }
    TrieNode root=new TrieNode();
    public void addword(String word)
    {
        TrieNode cur=root;
        for(int i=0;i<word.length();i++)
        {
            char c=word.charAt(i);
            if(cur.links[c-'a']==null)
            {
                cur.links[c-'a']=new TrieNode();
            }
            cur=cur.links[c-'a'];
        }
        cur.word=word;
        cur.isEnd=true;
    }
    
}
```



### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)



### [211. 添加与搜索单词 - 数据结构设计](https://leetcode-cn.com/problems/design-add-and-search-words-data-structure/)



### [648. 单词替换](https://leetcode-cn.com/problems/replace-words/)

首先将词典中的单词加入字典树tree

将句子以空格划分为单词数组words[]

对单词数组中的每个单词，在字典树中寻找最短的前缀（词根）

如果字典树中没有其前缀，结果字符串加上单词本身

如果字典树中有前缀，则返回其前缀

### [677. 键值映射](https://leetcode-cn.com/problems/map-sum-pairs/)

前缀树节点中包含links指向下面的节点，value表示到此为止的单词key对应的值val，sum表示以此为前缀的val之和，比如向树中插入<apple,5>，<ap,2>，则a对应的sum为7，a下面的p对应的sum为7，依此类推 

![image-20211025151247593](C:\Users\79937\AppData\Roaming\Typora\typora-user-images\image-20211025151247593.png)

```java
	class TrieNode{
        TrieNode[] links=new TrieNode[26];
        //boolean isEnd;
        int sum;
        int value;
    }
```

每次插入键值对，如果之前没有key（cur.value==0），则正常插入，每个字符对应的节点的sum都加上val

如果已经有key，  这条路径上对应的sum还需要减去之前的val，

```java
		public void insert(String key, int val) {
        TrieNode cur=root;
        for(char c:key.toCharArray())
        {
            if(cur.links[c-'a']==null)
            {
                cur.links[c-'a']=new TrieNode();
            }
            cur.links[c-'a'].sum+=val;
            cur=cur.links[c-'a'];
        }
        if(cur.value==0)
        {
            cur.value=val;
            //cur.sum=val;
        }
        else
        {
            int old=cur.value;
            cur=root;
            for(char c:key.toCharArray())
            {
                cur.links[c-'a'].sum-=old;
                cur=cur.links[c-'a'];
            }
            cur.value=val;
        }
    }

```

取前缀的sum就是搜索到对应的前缀prefix的最后一个字符，其sum就是结果

```java
		public int sum(String prefix) {
        TrieNode cur=root;
        for(char c:prefix.toCharArray())
        {
            if(cur.links[c-'a']==null)
            {
                //System.out.println(c);
                return 0;
            }
            cur=cur.links[c-'a'];
        }
        return cur.sum;
    }
```

也可以使用dfs遍历求val之和

```java
public int sum(String prefix) {
        TrieNode cur=root;
        for(char c:prefix.toCharArray())
        {
            if(cur.links[c-'a']==null)
            {
                return 0;
            }
            cur=cur.links[c-'a'];
        }
        return total(cur);
    }

    private int total(TrieNode cur)
    {
        int re=cur.val;
        for(TrieNode node:cur.links)
        {
            if(node!=null)
            {
                re+=total(node);
            }
        }
        return re;
    }
```



### [720. 词典中最长的单词](https://leetcode-cn.com/problems/longest-word-in-dictionary/)

字典树+dfs

将词典中的单词，加入字典树

深度遍历

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

  

## 洗牌算法

### [384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)

暴力法

随机生成n个不同的数组下标，并使用set来去重，时间复杂度为O(n^2)

```java
	public int[] shuffle() {
        Set<Integer> set=new HashSet<>();
        Random rd = new Random();
        int[] nums=new int[n];
        for(int i=0;i<n;i++){
            int index=rd.nextInt(n);
            while(!set.add(index)){
                index=rd.nextInt(n);
            }
            nums[i]=arr[index];
        }
        return nums;
    }
```

洗牌算法
共有 n 个不同的数，根据每个位置能够选择什么数，共有 n! 种组合。

题目要求每次调用 shuffle 时等概率返回某个方案，或者说每个元素都够等概率出现在每个位置中。

我们可以使用 Knuth 洗牌算法，在 O(n) 复杂度内等概率返回某个方案。

具体的，我们从前往后尝试填充 [0, n - 1][0,n−1] 该填入什么数时，通过随机当前下标与（剩余的）哪个下标进行值交换来实现。

对于下标 x 而言，我们从 [x, n - 1][x,n−1] 中随机出一个位置与 x 进行值交换，当所有位置都进行这样的处理后，我们便得到了一个公平的洗牌方案。

例如，对于下标为 0 位置，从 [0, n - 1][0,n−1] 随机一个位置进行交换，共有 n 种选择；下标为 1 的位置，从 [1, n - 1][1,n−1] 随机一个位置进行交换，共有 n−1 种选择 ... 且每个位置的随机位置交换过程相互独立。

```java
	public int[] shuffle() {
        Set<Integer> set=new HashSet<>();
        Random rd = new Random();
        int[] nums=(int[])Arrays.copyOf(arr,n);
        for(int i=0;i<n;i++){
            int j=rd.nextInt(n-i)+i;
            int temp=nums[i];
            nums[i]=nums[j];
            nums[j]=temp;
        }
        return nums;
    }
```

